
import re
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from scipy.optimize import milp, LinearConstraint, Bounds


# ═══════════════════════════════════════════════════════════════
# STAGE 0 — DATA MODELS (Canonical Schemas)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PaymentEvent:
    payment_id: str
    amount: float
    currency: str
    payment_date: str
    bank_account: str
    channel: str              # ACH, wire, check, lockbox, portal
    raw_remittance: str       # NEVER cleaned — held for extraction
    payer_name: str = ""
    payer_email: str = ""

@dataclass
class Invoice:
    invoice_id: str
    customer_id: str
    amount: float
    currency: str
    issue_date: str
    due_date: str
    status: str = "open"
    days_old: int = 0

@dataclass
class Customer:
    customer_id: str
    name: str
    bank_accounts: List[str] = field(default_factory=list)
    email_domain: str = ""
    aliases: List[str] = field(default_factory=list)

@dataclass
class RemittanceData:
    invoice_refs: List[str]
    amounts: List[float]
    deduction_hints: List[str]
    raw_text: str
    field_confidence: Dict[str, float] = field(default_factory=dict)

@dataclass
class MatchResult:
    payment_id: str
    customer_id: str
    allocations: List[Dict]
    total_allocated: float
    deviation: float
    engine_status: str
    confidence: float
    decision: str            # auto_post / review / manual / unapplied
    reasoning: List[str] = field(default_factory=list)

@dataclass
class FeedbackRecord:
    payment_id: str
    prediction: Dict
    features: Dict
    timestamp: str
    human_correction: Optional[Dict] = None


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — REMITTANCE EXTRACTION
# ═══════════════════════════════════════════════════════════════

class RemittanceExtractor:
    """
    Extract structured data from raw remittance text.
    Production: replace regex with OCR + LayoutLM + LLM.
    Current: regex baseline (fast, debuggable, no dependencies).
    """

    INVOICE_PATTERNS = [
        r'(INV[-\s]?\d+)',
        r'(Invoice\s*#?\s*\d+)',
        r'(inv\s*\d+)',
    ]

    AMOUNT_PATTERN = r'\$?([\d,]+\.?\d{0,2})'

    DEDUCTION_KEYWORDS = [
        'discount', 'promo', 'credit', 'return',
        'deduction', 'rebate', 'adjustment', 'less'
    ]

    def extract(self, raw_text: str) -> RemittanceData:
        text_lower = raw_text.lower()

        # Extract invoice references
        refs = []
        for pattern in self.INVOICE_PATTERNS:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            refs.extend([self._normalize_ref(m) for m in matches])
        refs = list(dict.fromkeys(refs))  # deduplicate, preserve order

        # Extract mentioned amounts
        amounts = []
        for match in re.findall(self.AMOUNT_PATTERN, raw_text):
            try:
                val = float(match.replace(',', ''))
                if val > 0:
                    amounts.append(val)
            except ValueError:
                pass

        # Detect deduction hints
        deductions = [kw for kw in self.DEDUCTION_KEYWORDS if kw in text_lower]

        # Confidence scoring per field
        confidence = {
            'invoice_refs': 0.95 if refs else 0.10,
            'amounts': 0.80 if amounts else 0.10,
            'deductions': 0.70 if deductions else 0.50,
        }

        return RemittanceData(
            invoice_refs=refs,
            amounts=amounts,
            deduction_hints=deductions,
            raw_text=raw_text,
            field_confidence=confidence
        )

    def _normalize_ref(self, ref: str) -> str:
        return re.sub(r'[\s]+', '-', ref.strip().upper())


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — CUSTOMER IDENTITY RESOLUTION
# ═══════════════════════════════════════════════════════════════

class CustomerResolver:
    """
    Resolve which ERP customer sent this payment.
    Uses: bank account (exact), name similarity (fuzzy), email domain.
    """

    def __init__(self, customers: List[Customer]):
        self.customers = customers

    def resolve(self, payment: PaymentEvent) -> Tuple[Optional[str], float, List[str]]:
        best_customer = None
        best_score = 0.0
        reasoning = []

        for cust in self.customers:
            score = 0.0
            reasons = []

            # Signal 1: Bank account (strongest — exact match)
            if payment.bank_account in cust.bank_accounts:
                score += 0.70
                reasons.append(f"Bank account exact match: {payment.bank_account}")

            # Signal 2: Name similarity (fuzzy)
            name_sim = self._name_similarity(payment.payer_name, cust.name, cust.aliases)
            score += 0.20 * name_sim
            if name_sim > 0.5:
                reasons.append(f"Name similarity: {name_sim:.2f}")

            # Signal 3: Email domain
            if payment.payer_email and cust.email_domain:
                if cust.email_domain in payment.payer_email.lower():
                    score += 0.10
                    reasons.append(f"Email domain match: {cust.email_domain}")

            if score > best_score:
                best_score = score
                best_customer = cust.customer_id
                reasoning = reasons

        return best_customer, min(best_score, 1.0), reasoning

    def _name_similarity(self, payer_name: str, cust_name: str, aliases: List[str]) -> float:
        """Simple character-level similarity (production: use Jaro-Winkler)"""
        all_names = [cust_name] + aliases
        best = 0.0
        for name in all_names:
            a, b = payer_name.lower(), name.lower()
            if not a or not b:
                continue
            common = sum(1 for c in a if c in b)
            sim = (2.0 * common) / (len(a) + len(b))
            best = max(best, sim)
        return best


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — CANDIDATE INVOICE RETRIEVAL
# ═══════════════════════════════════════════════════════════════

class CandidateRetriever:
    """
    Fetch candidate invoices. Optimizes for RECALL — include all
    plausible invoices so downstream scoring can filter.
    """

    def retrieve(self, customer_id: str, payment: PaymentEvent,
                 remittance: RemittanceData, all_invoices: List[Invoice],
                 max_candidates: int = 30) -> List[Invoice]:

        # Filter to customer's open invoices
        candidates = [
            inv for inv in all_invoices
            if inv.customer_id == customer_id and inv.status == "open"
        ]

        # Score each candidate for relevance (recall-oriented)
        scored = []
        for inv in candidates:
            relevance = 0.0

            # Reference mentioned in remittance
            if inv.invoice_id in remittance.invoice_refs:
                relevance += 5.0

            # Amount within proximity of payment
            if inv.amount <= payment.amount * 1.05:
                relevance += 1.0

            # Aging signal (older invoices more likely to be paid)
            relevance += min(inv.days_old / 90.0, 1.0)

            scored.append((inv, relevance))

        # Sort by relevance, return top candidates
        scored.sort(key=lambda x: x[1], reverse=True)
        return [inv for inv, _ in scored[:max_candidates]]


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Produce features for each (payment, invoice) pair.
    Four feature categories: amount, reference, temporal, behavioral.
    """

    def compute(self, payment: PaymentEvent, invoice: Invoice,
                remittance: RemittanceData, tolerance_pct: float = 0.02) -> Dict:

        # --- Amount features ---
        amount_diff = abs(payment.amount - invoice.amount)
        amount_ratio = invoice.amount / payment.amount if payment.amount > 0 else 0
        within_tolerance = 1.0 if amount_diff <= payment.amount * tolerance_pct else 0.0

        # --- Reference features ---
        exact_ref = 1.0 if invoice.invoice_id in remittance.invoice_refs else 0.0
        fuzzy_ref = self._fuzzy_ref_score(invoice.invoice_id, remittance.raw_text)

        # --- Temporal features ---
        days_past_due = invoice.days_old
        is_overdue = 1.0 if days_past_due > 30 else 0.0

        # --- Composite text score (for MILP) ---
        text_score = max(exact_ref, fuzzy_ref)

        return {
            'amount_diff': amount_diff,
            'amount_ratio': amount_ratio,
            'within_tolerance': within_tolerance,
            'exact_ref_match': exact_ref,
            'fuzzy_ref_score': fuzzy_ref,
            'days_past_due': days_past_due,
            'is_overdue': is_overdue,
            'text_score': text_score,
        }

    def _fuzzy_ref_score(self, invoice_id: str, text: str) -> float:
        """Substring fuzzy match score"""
        inv_clean = invoice_id.lower().replace('-', '').replace(' ', '')
        text_clean = text.lower().replace('-', '').replace(' ', '')
        if inv_clean in text_clean:
            return 1.0
        # Partial numeric match
        nums = re.findall(r'\d+', invoice_id)
        for num in nums:
            if num in text_clean:
                return 0.6
        return 0.0


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — ML SCORING MODEL
# ═══════════════════════════════════════════════════════════════

class MLScorer:
    """
    Score each candidate invoice.
    Current: weighted feature combination (rule-based baseline).
    Production upgrade: plug in trained XGBoost model.
    """

    # Feature weights (learned from domain expertise, replace with XGBoost)
    WEIGHTS = {
        'exact_ref_match': 0.40,
        'fuzzy_ref_score': 0.15,
        'amount_ratio': 0.20,
        'within_tolerance': 0.10,
        'is_overdue': 0.05,
        'days_past_due': 0.10,  # normalized
    }

    def score(self, features: Dict) -> float:
        score = 0.0
        score += self.WEIGHTS['exact_ref_match'] * features['exact_ref_match']
        score += self.WEIGHTS['fuzzy_ref_score'] * features['fuzzy_ref_score']
        score += self.WEIGHTS['amount_ratio'] * min(features['amount_ratio'], 1.0)
        score += self.WEIGHTS['within_tolerance'] * features['within_tolerance']
        score += self.WEIGHTS['is_overdue'] * features['is_overdue']

        # Normalize days_past_due to 0-1 range
        dpd_norm = min(features['days_past_due'] / 90.0, 1.0)
        score += self.WEIGHTS['days_past_due'] * dpd_norm

        return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — MILP OPTIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════

class MILPOptimizer:
    """
    Find optimal invoice subset using Mixed-Integer Linear Programming.

    Variables:
        x_i ∈ {0,1}  — include invoice i
        s+, s-       — positive/negative slack (deviation)

    Objective (minimize):
        -(amounts + alpha * text_scores) * x + penalty * (s+ + s-)

    Constraint:
        sum(amount_i * x_i) + s+ - s- = payment_amount
    """

    def __init__(self, alpha: float = 0.5, penalty: float = 5.0):
        self.alpha = alpha
        self.penalty = penalty

    def solve(self, payment_amount: float, candidates: List[Invoice],
              feature_map: Dict[str, Dict]) -> Tuple[str, List[Dict], float]:

        amounts = np.array([inv.amount for inv in candidates])
        text_scores = np.array([
            feature_map[inv.invoice_id]['text_score'] for inv in candidates
        ])
        num_inv = len(amounts)

        if num_inv == 0:
            return "No Candidates", [], 0.0

        # --- Build MILP ---
        total_vars = num_inv + 2  # invoices + slack_pos + slack_neg

        # Objective: maximize (amounts + text_signal) — minimize deviation
        c = np.concatenate([
            -(amounts + self.alpha * text_scores),
            [self.penalty, self.penalty]
        ])

        # Constraint: sum(a_i * x_i) + s+ - s- = payment
        A = np.zeros((1, total_vars))
        A[0, :num_inv] = amounts
        A[0, num_inv] = 1       # slack_pos
        A[0, num_inv + 1] = -1  # slack_neg

        constraint = LinearConstraint(A, lb=payment_amount, ub=payment_amount)

        # Variable types
        integrality = np.concatenate([
            np.ones(num_inv),
            [0, 0]
        ])

        bounds = Bounds(
            [0] * num_inv + [0, 0],
            [1] * num_inv + [np.inf, np.inf]
        )

        # --- Solve ---
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)

        if not res.success or res.x is None:
            return "MILP Failed", [], 0.0

        # --- Parse results ---
        x = res.x[:num_inv]
        slack_pos = res.x[num_inv]
        slack_neg = res.x[num_inv + 1]
        deviation = abs(slack_pos - slack_neg)

        selected = np.where(x > 0.5)[0]
        total = sum(amounts[i] for i in selected)

        allocations = []
        for idx in selected:
            inv = candidates[idx]
            allocations.append({
                'invoice_id': inv.invoice_id,
                'allocated_amount': inv.amount,
                'text_score': float(text_scores[idx]),
            })

        # Status classification
        if deviation <= 0.01:
            if len(selected) > 1:
                status = "Exact Multi-Invoice Match"
            elif len(selected) == 1:
                status = "Exact Single Match"
            else:
                status = "No Match Found"
        else:
            status = f"Match with Deviation ({deviation:.2f})"

        return status, allocations, deviation


# ═══════════════════════════════════════════════════════════════
# STAGE 7 — CONFIDENCE GATING & DECISION ENGINE
# ═══════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Convert match results into routing decisions.
    Thresholds tunable per customer/segment.
    """

    def __init__(self, auto_threshold=0.92, review_threshold=0.60):
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

    def decide(self, status: str, allocations: List[Dict],
               deviation: float, payment_amount: float,
               feature_map: Dict) -> Tuple[float, str, List[str]]:

        reasoning = []

        if not allocations:
            return 0.0, "unapplied", ["No matching invoices found"]

        # --- Compute composite confidence ---
        total_allocated = sum(a['allocated_amount'] for a in allocations)

        # Factor 1: Amount accuracy
        amount_accuracy = 1.0 - (deviation / payment_amount) if payment_amount > 0 else 0
        reasoning.append(f"Amount accuracy: {amount_accuracy:.3f}")

        # Factor 2: Text support
        text_support = np.mean([a['text_score'] for a in allocations])
        reasoning.append(f"Text support: {text_support:.3f}")

        # Factor 3: Match completeness
        completeness = min(total_allocated / payment_amount, 1.0) if payment_amount > 0 else 0
        reasoning.append(f"Completeness: {completeness:.3f}")

        # Weighted confidence
        confidence = (
            0.50 * amount_accuracy +
            0.30 * text_support +
            0.20 * completeness
        )
        confidence = min(max(confidence, 0.0), 1.0)

        # --- Route ---
        if confidence >= self.auto_threshold:
            decision = "auto_post"
        elif confidence >= self.review_threshold:
            decision = "review"
        else:
            decision = "manual"

        reasoning.append(f"Final confidence: {confidence:.3f}")
        reasoning.append(f"Decision: {decision}")

        return confidence, decision, reasoning


# ═══════════════════════════════════════════════════════════════
# STAGE 8 — FEEDBACK LOGGER
# ═══════════════════════════════════════════════════════════════

class FeedbackLogger:
    """
    Log predictions for future model training.
    Every human correction becomes a labeled training example.
    """

    def __init__(self):
        self.records: List[FeedbackRecord] = []

    def log(self, result: MatchResult, feature_map: Dict) -> FeedbackRecord:
        record = FeedbackRecord(
            payment_id=result.payment_id,
            prediction={
                'allocations': result.allocations,
                'confidence': result.confidence,
                'decision': result.decision,
                'status': result.engine_status,
            },
            features=feature_map,
            timestamp=datetime.now().isoformat(),
        )
        self.records.append(record)
        return record


# ═══════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class CashApplicationPipeline:
    """
    Full production pipeline orchestrator.
    Connects all stages in sequence.
    """

    def __init__(self, customers: List[Customer], invoices: List[Invoice]):
        self.extractor = RemittanceExtractor()
        self.resolver = CustomerResolver(customers)
        self.retriever = CandidateRetriever()
        self.fe = FeatureEngineer()
        self.scorer = MLScorer()
        self.optimizer = MILPOptimizer(alpha=0.5, penalty=5.0)
        self.decision_engine = DecisionEngine()
        self.feedback = FeedbackLogger()
        self.invoices = invoices

    def process(self, payment: PaymentEvent) -> MatchResult:
        print("=" * 70)
        print(f"  PROCESSING PAYMENT: {payment.payment_id}")
        print(f"  Amount: {payment.currency} {payment.amount:,.2f}")
        print("=" * 70)

        # --- STAGE 1: Remittance Extraction ---
        remittance = self.extractor.extract(payment.raw_remittance)
        print(f"\n[STAGE 1] REMITTANCE EXTRACTION")
        print(f"  Invoice refs found  : {remittance.invoice_refs}")
        print(f"  Amounts found       : {remittance.amounts}")
        print(f"  Deduction hints     : {remittance.deduction_hints}")
        print(f"  Field confidence    : {remittance.field_confidence}")

        # --- STAGE 2: Customer Resolution ---
        customer_id, cust_conf, cust_reasons = self.resolver.resolve(payment)
        print(f"\n[STAGE 2] CUSTOMER RESOLUTION")
        print(f"  Resolved customer   : {customer_id}")
        print(f"  Confidence          : {cust_conf:.3f}")
        for r in cust_reasons:
            print(f"    → {r}")

        if customer_id is None or cust_conf < 0.50:
            print(f"  ⚠ Customer resolution failed — routing to manual")
            return MatchResult(
                payment_id=payment.payment_id, customer_id="UNKNOWN",
                allocations=[], total_allocated=0, deviation=0,
                engine_status="Customer Unresolved", confidence=0,
                decision="manual", reasoning=["Customer resolution below threshold"]
            )

        # --- STAGE 3: Candidate Retrieval ---
        candidates = self.retriever.retrieve(customer_id, payment, remittance, self.invoices)
        print(f"\n[STAGE 3] CANDIDATE RETRIEVAL")
        print(f"  Candidates retrieved: {len(candidates)}")
        for inv in candidates:
            print(f"    → {inv.invoice_id}: {payment.currency} {inv.amount:,.2f} ({inv.days_old}d old)")

        # --- STAGE 4: Feature Engineering ---
        feature_map = {}
        for inv in candidates:
            features = self.fe.compute(payment, inv, remittance)
            feature_map[inv.invoice_id] = features
        print(f"\n[STAGE 4] FEATURE ENGINEERING")
        for inv_id, feats in feature_map.items():
            ref_signal = "✓ REF" if feats['exact_ref_match'] > 0 else "  ---"
            print(f"    {inv_id}: diff={feats['amount_diff']:>8,.0f}  "
                  f"ratio={feats['amount_ratio']:.2f}  "
                  f"{ref_signal}  "
                  f"text={feats['text_score']:.1f}  "
                  f"dpd={feats['days_past_due']:>3d}")

        # --- STAGE 5: ML Scoring ---
        ml_scores = {}
        for inv in candidates:
            ml_scores[inv.invoice_id] = self.scorer.score(feature_map[inv.invoice_id])
        print(f"\n[STAGE 5] ML SCORING (candidate ranking)")
        sorted_scores = sorted(ml_scores.items(), key=lambda x: x[1], reverse=True)
        for inv_id, score in sorted_scores:
            bar = "█" * int(score * 30)
            print(f"    {inv_id}: {score:.3f}  {bar}")

        # --- STAGE 6: MILP Optimization ---
        status, allocations, deviation = self.optimizer.solve(
            payment.amount, candidates, feature_map
        )
        print(f"\n[STAGE 6] MILP OPTIMIZATION")
        print(f"  Status              : {status}")
        print(f"  Deviation           : {deviation:.2f}")
        total_alloc = sum(a['allocated_amount'] for a in allocations) if allocations else 0
        print(f"  Total allocated     : {payment.currency} {total_alloc:,.2f}")
        for a in allocations:
            print(f"    → {a['invoice_id']}: {payment.currency} {a['allocated_amount']:,.2f}"
                  f"  (text={a['text_score']:.1f})")

        # --- STAGE 7: Decision Engine ---
        confidence, decision, reasoning = self.decision_engine.decide(
            status, allocations, deviation, payment.amount, feature_map
        )
        print(f"\n[STAGE 7] CONFIDENCE GATING & DECISION")
        for r in reasoning:
            print(f"    → {r}")

        decision_icons = {
            "auto_post": "✅ AUTO-POST",
            "review": "🔍 REVIEW QUEUE",
            "manual": "⚠️  MANUAL",
            "unapplied": "❌ UNAPPLIED"
        }
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  DECISION: {decision_icons.get(decision, decision):<27}║")
        print(f"  ║  CONFIDENCE: {confidence:.3f}                    ║")
        print(f"  ╚══════════════════════════════════════╝")

        # --- Build result ---
        result = MatchResult(
            payment_id=payment.payment_id,
            customer_id=customer_id,
            allocations=allocations,
            total_allocated=total_alloc,
            deviation=deviation,
            engine_status=status,
            confidence=confidence,
            decision=decision,
            reasoning=reasoning
        )

        # --- STAGE 8: Feedback ---
        record = self.feedback.log(result, feature_map)
        print(f"\n[STAGE 8] FEEDBACK LOGGED")
        print(f"  Record ID           : {record.payment_id}")
        print(f"  Timestamp           : {record.timestamp}")
        print(f"  Features logged     : {len(feature_map)} invoice pairs")

        print("\n" + "=" * 70)
        return result


# ═══════════════════════════════════════════════════════════════
# DEMO — FULL PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════

def run_demo():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     PRODUCTION CASH APPLICATION MATCHING PIPELINE              ║")
    print("║     Version 1.0 — Hybrid ML + Optimization Engine             ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    # --- Setup: Customers ---
        # =========================================================================
    # TEST DATA: CUSTOMERS (Including aliases, bank accounts, and email domains)
    # =========================================================================
    customers = [
        Customer(
            customer_id="CUST-001",
            name="Acme East Africa",
            bank_accounts=["KE123456", "KE789012"],
            email_domain="acme-ea.co.ke",
            aliases=["Acme EA", "AEA Holdings"]
        ),
        Customer(
            customer_id="CUST-002",
            name="Nairobi Supplies Ltd",
            bank_accounts=["KE555555"],
            email_domain="nairobisupplies.co.ke",
            aliases=["Nairobi Supplies"]
        ),
        Customer(
            customer_id="CUST-003",
            name="Kilimanjaro Logistics",
            bank_accounts=["TZ999888", "KE888111"],
            email_domain="kililogistics.com",
            aliases=["Kili Log", "Kilimanjaro Exp"]
        ),
        Customer(
            customer_id="CUST-004",
            name="Rift Valley Distributors",
            bank_accounts=["KE444333"],
            email_domain="rvdistributors.co.ke",
            aliases=["RVD Ltd", "Rift Valley Dist"]
        ),
        Customer(
            customer_id="CUST-005",
            name="Safari Tech Solutions",
            bank_accounts=["KE222111"],
            email_domain="safaritech.io",
            aliases=["SafariTech"]
        )
    ]

    # =========================================================================
    # TEST DATA: OPEN INVOICES (Varying outstanding totals and aging profiles)
    # =========================================================================
    invoices = [
        # Customer 001 Open Ledger (Acme East Africa)
        Invoice("INV-101", "CUST-001", 5000.00, "KES", "2026-04-01", "2026-05-01", "open", 66),
        Invoice("INV-102", "CUST-001", 3000.00, "KES", "2026-04-15", "2026-05-15", "open", 52),
        Invoice("INV-103", "CUST-001", 2000.00, "KES", "2026-05-01", "2026-05-31", "open", 36),
        Invoice("INV-104", "CUST-001", 7000.00, "KES", "2026-05-10", "2026-06-10", "open", 27),
        Invoice("INV-105", "CUST-001", 1500.00, "KES", "2026-05-20", "2026-06-20", "open", 17),
        Invoice("INV-106", "CUST-001", 4500.00, "KES", "2026-05-25", "2026-06-25", "open", 12),

        # Customer 002 Open Ledger (Nairobi Supplies Ltd)
        Invoice("INV-201", "CUST-002", 12500.00, "KES", "2026-03-10", "2026-04-10", "open", 88),
        Invoice("INV-202", "CUST-002", 8400.00, "KES", "2026-04-20", "2026-05-20", "open", 47),
        Invoice("INV-203", "CUST-002", 3100.00, "KES", "2026-05-15", "2026-06-15", "open", 22),

        # Customer 003 Open Ledger (Kilimanjaro Logistics)
        Invoice("INV-301", "CUST-003", 4500.00, "USD", "2026-02-01", "2026-03-01", "open", 125),
        Invoice("INV-302", "CUST-003", 6200.00, "USD", "2026-05-05", "2026-06-05", "open", 32),

        # Customer 004 Open Ledger (Rift Valley Distributors)
        Invoice("INV-401", "CUST-004", 15000.00, "KES", "2026-05-01", "2026-05-31", "open", 36),
        Invoice("INV-402", "CUST-004", 15000.00, "KES", "2026-05-12", "2026-06-12", "open", 25),

        # Customer 005 Open Ledger (Safari Tech Solutions)
        Invoice("INV-501", "CUST-005", 9800.00, "KES", "2026-05-18", "2026-06-18", "open", 19)
    ]

    # =========================================================================
    # TEST DATA: INCOMING PAYMENTS (Scenarios mapping to different strategies)
    # =========================================================================
    payments = [
        # CASE 1: Perfect Multi-Invoice Match via Total Sum & Text References
        # Strategy Map: Subset-Sum optimization confirms exactly 5000 + 3000 + 2000 = 10000
        PaymentEvent(
            payment_id="PAY-2026-0042",
            amount=10000.00,
            currency="KES",
            payment_date="2026-06-06",
            bank_account="KE123456",
            channel="ACH",
            raw_remittance="Payment for INV-101 and INV-102 and INV-103",
            payer_name="Acme East Africa",
            payer_email="accounts@acme-ea.co.ke"
        ),

        # CASE 2: No Invoice Numbers Provided, but Unique Inverse Amount Match Found
        # Strategy Map: CUST-002 only has one invoice matching 8400.00 exactly (INV-202)
        PaymentEvent(
            payment_id="PAY-2026-0043",
            amount=8400.00,
            currency="KES",
            payment_date="2026-06-07",
            bank_account="KE555555",
            channel="RTGS",
            raw_remittance="WIRE TRF FROM NAIROBI SUPPLIES",
            payer_name="Nairobi Supplies",
            payer_email="finance@nairobisupplies.co.ke"
        ),

        # CASE 3: Clean Multi-Invoice Subset-Sum Puzzle (No Memo Clues)
        # Strategy Map: CUST-002 sends 15600.00. Solver finds INV-201 (12500) + INV-203 (3100) = 15600
        PaymentEvent(
            payment_id="PAY-2026-0044",
            amount=15600.00,
            currency="KES",
            payment_date="2026-06-07",
            bank_account="KE555555",
            channel="ACH",
            raw_remittance="BULK INVOICE SETTLEMENT",
            payer_name="Nairobi Supplies Ltd",
            payer_email="ap@nairobisupplies.co.ke"
        ),

        # CASE 4: Small Short-Payment / Tolerance Threshold Variance Test
        # Strategy Map: Customer pays 9650.00 on a 9800.00 invoice (INV-501). 
        # Falls within a 2% variance limit ($150 diff). Triggers write-off pipeline.
        PaymentEvent(
            payment_id="PAY-2026-0045",
            amount=9650.00,
            currency="KES",
            payment_date="2026-06-08",
            bank_account="KE222111",
            channel="MPESA",
            raw_remittance="SAFARI-TECH INV-501 LESS CHARGES",
            payer_name="SafariTech",
            payer_email="accounting@safaritech.io"
        ),

        # CASE 5: Duplicate Inverse Amounts (Forces Fallback to Text or FIFO)
        # Strategy Map: Matrix solver finds two identical $15,000 invoices (INV-401 & INV-402).
        # Engine must evaluate the text score to select INV-402, skipping the older one.
        PaymentEvent(
            payment_id="PAY-2026-0046",
            amount=15000.00,
            currency="KES",
            payment_date="2026-06-08",
            bank_account="KE444333",
            channel="RTGS",
            raw_remittance="SETTLEMENT FOR PARTS - REF: INV-402",
            payer_name="Rift Valley Dist",
            payer_email="payables@rvdistributors.co.ke"
        ),

        # CASE 6: Total Blind Payment (Forces FIFO Breakdown Rule)
        # Strategy Map: No subset calculation matches 11000.00. 
        # Engine must sort CUST-001 by days_old and chip away: INV-101 (5000), INV-102 (3000), then remaining 3000 to INV-103.
        PaymentEvent(
            payment_id="PAY-2026-0047",
            amount=11000.00,
            currency="KES",
            payment_date="2026-06-08",
            bank_account="KE789012",
            channel="ACH",
            raw_remittance="ON ACCOUNT DEPOSIT AEA HOLDINGS",
            payer_name="AEA Holdings",
            payer_email="treasury@acme-ea.co.ke"
        )
    ]


    # --- Run Pipeline ---
    pipeline = CashApplicationPipeline(customers, invoices)
    for payment in payments:

        result = pipeline.process(payment)

        # --- Final Summary ---
        print("\n\n" + "═" * 70)
        print("  PIPELINE RESULT SUMMARY")
        print("═" * 70)
        print(f"  Payment      : {result.payment_id}")
        print(f"  Customer     : {result.customer_id}")
        print(f"  Status       : {result.engine_status}")
        print(f"  Allocated    : KES {result.total_allocated:,.2f}")
        print(f"  Deviation    : {result.deviation:.2f}")
        print(f"  Confidence   : {result.confidence:.3f}")
        print(f"  Decision     : {result.decision.upper()}")
        print(f"  Invoices     :")
        for a in result.allocations:
            print(f"    → {a['invoice_id']}: KES {a['allocated_amount']:,.2f}")
        print("═" * 70)

    # --- Show feedback records ---
    print(f"\n  Feedback records stored: {len(pipeline.feedback.records)}")
    print(f"  Ready for XGBoost training: ✓")
    print()


run_demo()

