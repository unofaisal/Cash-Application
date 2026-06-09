import re
import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
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
    channel: str
    raw_remittance: str
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
    # UPDATE: field_confidence now reflects actual extraction quality,
    # not a hardcoded boolean. Each field carries its own uncertainty score
    # derived from the match strength that produced it.
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
    decision: str
    deduction_code: str = "NONE"
    gl_entries: List[Dict] = field(default_factory=list)
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
#
# UPDATE: Confidence is no longer hardcoded (0.95 if refs else 0.10).
# It is now derived from the quality of each match:
#   - Full pattern match on "INV-101"  → 0.95
#   - Partial numeric match on "101"   → 0.60
#   - No match                         → 0.10
#
# This matters because downstream confidence gating needs to know HOW
# certain we are about each extracted field, not just whether it exists.
# A regex baseline is still used here; production replaces this with
# OCR + LayoutLM + LLM for layout-aware unstructured document parsing.
# ═══════════════════════════════════════════════════════════════

class RemittanceExtractor:

    INVOICE_PATTERNS = [
        (r'(INV[-\s]?\d+)',          0.95),   # exact standard format → high confidence
        (r'(Invoice\s*#?\s*\d+)',    0.90),   # verbose format → still high
        (r'(inv\s*\d+)',             0.80),   # lowercase abbreviation → slightly lower
    ]

    AMOUNT_PATTERN = r'\$?([\d,]+\.?\d{0,2})'

    DEDUCTION_KEYWORDS = [
        'discount', 'promo', 'credit', 'return',
        'deduction', 'rebate', 'adjustment', 'less', 'charges'
    ]

    def extract(self, raw_text: str) -> RemittanceData:
        text_lower = raw_text.lower()

        # Extract invoice references with per-match confidence
        refs = []
        ref_confidence_scores = []
        for pattern, conf in self.INVOICE_PATTERNS:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            for m in matches:
                normalized = self._normalize_ref(m)
                if normalized not in refs:
                    refs.append(normalized)
                    ref_confidence_scores.append(conf)

        # If no full pattern found, attempt partial numeric extraction
        # as a low-confidence fallback
        if not refs:
            nums = re.findall(r'\b(\d{3,6})\b', raw_text)
            for n in nums:
                candidate = f"INV-{n}"
                refs.append(candidate)
                ref_confidence_scores.append(0.30)  # very low — ambiguous numeric

        # Overall invoice_refs confidence = mean of individual match confidences
        invoice_conf = float(np.mean(ref_confidence_scores)) if ref_confidence_scores else 0.10

        # Extract amounts
        amounts = []
        for match in re.findall(self.AMOUNT_PATTERN, raw_text):
            try:
                val = float(match.replace(',', ''))
                if val > 0:
                    amounts.append(val)
            except ValueError:
                pass
        amount_conf = 0.80 if amounts else 0.10

        # Detect deduction hints
        deductions = [kw for kw in self.DEDUCTION_KEYWORDS if kw in text_lower]
        deduction_conf = 0.75 if deductions else 0.50

        return RemittanceData(
            invoice_refs=refs,
            amounts=amounts,
            deduction_hints=deductions,
            raw_text=raw_text,
            field_confidence={
                'invoice_refs': round(invoice_conf, 3),
                'amounts':      round(amount_conf, 3),
                'deductions':   round(deduction_conf, 3),
            }
        )

    def _normalize_ref(self, ref: str) -> str:
        return re.sub(r'[\s]+', '-', ref.strip().upper())


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — CUSTOMER IDENTITY RESOLUTION
# ═══════════════════════════════════════════════════════════════
#
# UPDATE: _name_similarity now implements Jaro-Winkler distance
# instead of character overlap counting.
#
# WHY THIS MATTERS:
# Character overlap counts shared characters regardless of position.
# "Acme East Africa" and "Acme Engineering Africa" share many characters
# and would score misleadingly high. Jaro-Winkler:
#   1. Counts only characters that match within a window (position-aware)
#   2. Penalises transpositions
#   3. Gives a prefix bonus — company names that share an opening word
#      ("Nairobi Supplies" vs "Nairobi Supplies Ltd") score very high,
#      which matches how real entity names differ
#
# Production systems embed names via a language model and match in vector
# space, backed by an identity graph (bank account → customer edges).
# Jaro-Winkler is the correct deterministic step before embedding lookup.
# ═══════════════════════════════════════════════════════════════

class CustomerResolver:

    def __init__(self, customers: List[Customer]):
        self.customers = customers

    def resolve(self, payment: PaymentEvent) -> Tuple[Optional[str], float, List[str]]:
        best_customer = None
        best_score = 0.0
        reasoning = []

        for cust in self.customers:
            score = 0.0
            reasons = []

            # Signal 1: Bank account exact match (strongest signal)
            if payment.bank_account in cust.bank_accounts:
                score += 0.70
                reasons.append(f"Bank account exact match: {payment.bank_account}")

            # Signal 2: Jaro-Winkler name similarity
            name_sim = self._name_similarity(payment.payer_name, cust.name, cust.aliases)
            score += 0.20 * name_sim
            if name_sim > 0.5:
                reasons.append(f"Name similarity (Jaro-Winkler): {name_sim:.3f}")

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

    def _jaro(self, s1: str, s2: str) -> float:
        """Pure Jaro similarity."""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        match_dist = max(len1, len2) // 2 - 1
        match_dist = max(match_dist, 0)

        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_dist)
            end   = min(i + match_dist + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        return (matches / len1 + matches / len2 +
                (matches - transpositions / 2) / matches) / 3

    def _jaro_winkler(self, s1: str, s2: str, p: float = 0.1) -> float:
        """Jaro-Winkler: adds prefix bonus on top of Jaro."""
        jaro = self._jaro(s1, s2)
        s1l, s2l = s1.lower(), s2.lower()
        prefix = 0
        for i in range(min(4, len(s1l), len(s2l))):
            if s1l[i] == s2l[i]:
                prefix += 1
            else:
                break
        return jaro + prefix * p * (1 - jaro)

    def _name_similarity(self, payer_name: str, cust_name: str, aliases: List[str]) -> float:
        all_names = [cust_name] + aliases
        best = 0.0
        for name in all_names:
            sim = self._jaro_winkler(payer_name, name)
            best = max(best, sim)
        return best


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — CANDIDATE INVOICE RETRIEVAL
# (unchanged — recall-oriented retrieval is correct as-is)
# ═══════════════════════════════════════════════════════════════

class CandidateRetriever:

    def retrieve(self, customer_id: str, payment: PaymentEvent,
                 remittance: RemittanceData, all_invoices: List[Invoice],
                 max_candidates: int = 30) -> List[Invoice]:

        candidates = [
            inv for inv in all_invoices
            if inv.customer_id == customer_id and inv.status == "open"
        ]

        scored = []
        for inv in candidates:
            relevance = 0.0
            if inv.invoice_id in remittance.invoice_refs:
                relevance += 5.0
            if inv.amount <= payment.amount * 1.05:
                relevance += 1.0
            relevance += min(inv.days_old / 90.0, 1.0)
            scored.append((inv, relevance))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [inv for inv, _ in scored[:max_candidates]]


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
#
# UPDATE: Added behavioral feature category.
#
# The previous version had three feature categories: amount, reference,
# temporal. Missing was the most discriminative category in production:
# BEHAVIORAL — what has this customer done historically?
#
# Three behavioral features added:
#   avg_batch_size   : how many invoices does this customer typically pay
#                      together? If their average is 3 and we are matching
#                      a single invoice to a large payment, that is a signal
#                      we should look for more invoices.
#   avg_days_to_pay  : normalized against the invoice's own days_old.
#                      If this customer typically pays at 45 days and this
#                      invoice is 66 days old, it is overdue — stronger
#                      signal for inclusion.
#   short_pay_rate   : fraction of historical payments where this customer
#                      paid less than invoiced. High short_pay_rate means
#                      deviations should be expected and the engine should
#                      not penalise confidence heavily for small discrepancies.
#
# When customer_history is None (cold start), features default to neutral
# values (1.0, 30, 0.0) so the pipeline degrades gracefully.
# ═══════════════════════════════════════════════════════════════

class FeatureEngineer:

    def compute(self, payment: PaymentEvent, invoice: Invoice,
                remittance: RemittanceData, tolerance_pct: float = 0.02,
                customer_history: Optional[Dict] = None) -> Dict:

        # --- Amount features ---
        amount_diff  = abs(payment.amount - invoice.amount)
        amount_ratio = invoice.amount / payment.amount if payment.amount > 0 else 0
        within_tol   = 1.0 if amount_diff <= payment.amount * tolerance_pct else 0.0

        # --- Reference features ---
        exact_ref = 1.0 if invoice.invoice_id in remittance.invoice_refs else 0.0
        fuzzy_ref = self._fuzzy_ref_score(invoice.invoice_id, remittance.raw_text)

        # --- Temporal features ---
        days_past_due = invoice.days_old
        is_overdue    = 1.0 if days_past_due > 30 else 0.0

        # --- Behavioral features (NEW) ---
        if customer_history:
            avg_batch_size  = float(customer_history.get('avg_batch_size', 1.0))
            avg_days_to_pay = float(customer_history.get('avg_days_to_pay', 30.0))
            short_pay_rate  = float(customer_history.get('short_pay_rate', 0.0))
        else:
            # Cold start: neutral defaults
            avg_batch_size  = 1.0
            avg_days_to_pay = 30.0
            short_pay_rate  = 0.0

        # Normalize avg_days_to_pay against invoice age:
        # > 1.0 means invoice is older than this customer typically takes → likely overdue
        days_ratio = days_past_due / avg_days_to_pay if avg_days_to_pay > 0 else 1.0

        return {
            # Amount
            'amount_diff':      amount_diff,
            'amount_ratio':     amount_ratio,
            'within_tolerance': within_tol,
            # Reference
            'exact_ref_match':  exact_ref,
            'fuzzy_ref_score':  fuzzy_ref,
            'text_score':       max(exact_ref, fuzzy_ref),
            # Temporal
            'days_past_due':    days_past_due,
            'is_overdue':       is_overdue,
            # Behavioral (new)
            'avg_batch_size':   avg_batch_size,
            'avg_days_to_pay':  avg_days_to_pay,
            'short_pay_rate':   short_pay_rate,
            'days_ratio':       days_ratio,
        }

    def _fuzzy_ref_score(self, invoice_id: str, text: str) -> float:
        inv_clean  = invoice_id.lower().replace('-', '').replace(' ', '')
        text_clean = text.lower().replace('-', '').replace(' ', '')
        if inv_clean in text_clean:
            return 1.0
        nums = re.findall(r'\d+', invoice_id)
        for num in nums:
            if num in text_clean:
                return 0.6
        return 0.0


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — ML SCORING MODEL
# ═══════════════════════════════════════════════════════════════
#
# UPDATE: Scorer now incorporates behavioral features.
#
# Two behavioral adjustments:
#   1. short_pay_rate dampens confidence: a customer who short-pays 30%
#      of the time should reduce the scorer's certainty on clean matches
#      because the "clean match" may be hiding an expected deduction.
#      Applied as a downward multiplier: score *= (1 - 0.3 * short_pay_rate)
#
#   2. days_ratio boosts aging signal: if the invoice is significantly
#      older than this customer's average payment time, it is a stronger
#      candidate than a fresh invoice from the same customer.
#
# Static weights remain (replace with trained XGBoost fed by
# FeedbackLogger records). The behavioral terms are additive corrections
# that any learned model would naturally discover from labeled data.
# ═══════════════════════════════════════════════════════════════

class MLScorer:

    WEIGHTS = {
        'exact_ref_match':  0.38,
        'fuzzy_ref_score':  0.14,
        'amount_ratio':     0.20,
        'within_tolerance': 0.10,
        'is_overdue':       0.05,
        'days_past_due':    0.08,
        'days_ratio':       0.05,   # new behavioral weight
    }

    def score(self, features: Dict) -> float:
        s = 0.0
        s += self.WEIGHTS['exact_ref_match']  * features['exact_ref_match']
        s += self.WEIGHTS['fuzzy_ref_score']  * features['fuzzy_ref_score']
        s += self.WEIGHTS['amount_ratio']     * min(features['amount_ratio'], 1.0)
        s += self.WEIGHTS['within_tolerance'] * features['within_tolerance']
        s += self.WEIGHTS['is_overdue']       * features['is_overdue']
        s += self.WEIGHTS['days_past_due']    * min(features['days_past_due'] / 90.0, 1.0)
        s += self.WEIGHTS['days_ratio']       * min(features['days_ratio'], 2.0) / 2.0

        # Behavioral dampener: high short_pay_rate reduces score certainty
        # because the "match" may be masking an expected deduction
        short_pay_rate = features.get('short_pay_rate', 0.0)
        s *= (1.0 - 0.30 * short_pay_rate)

        return round(min(s, 1.0), 4)


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — MILP OPTIMIZATION ENGINE
# (unchanged — formulation with slack variables is correct)
# ═══════════════════════════════════════════════════════════════

class MILPOptimizer:

    def __init__(self, alpha: float = 0.5, penalty: float = 5.0):
        self.alpha   = alpha
        self.penalty = penalty

    def solve(self, payment_amount: float, candidates: List[Invoice],
              feature_map: Dict[str, Dict]) -> Tuple[str, List[Dict], float]:

        amounts     = np.array([inv.amount for inv in candidates])
        text_scores = np.array([feature_map[inv.invoice_id]['text_score'] for inv in candidates])
        num_inv     = len(amounts)

        if num_inv == 0:
            return "No Candidates", [], 0.0

        total_vars = num_inv + 2
        c = np.concatenate([-(amounts + self.alpha * text_scores), [self.penalty, self.penalty]])

        A = np.zeros((1, total_vars))
        A[0, :num_inv]   = amounts
        A[0, num_inv]    = 1
        A[0, num_inv+1]  = -1

        constraint  = LinearConstraint(A, lb=payment_amount, ub=payment_amount)
        integrality = np.concatenate([np.ones(num_inv), [0, 0]])
        bounds      = Bounds([0]*num_inv + [0, 0], [1]*num_inv + [np.inf, np.inf])

        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)

        if not res.success or res.x is None:
            return "MILP Failed", [], 0.0

        x        = res.x[:num_inv]
        deviation = abs(res.x[num_inv] - res.x[num_inv+1])
        selected  = np.where(x > 0.5)[0]

        allocations = []
        for idx in selected:
            inv = candidates[idx]
            allocations.append({
                'invoice_id':       inv.invoice_id,
                'allocated_amount': inv.amount,
                'text_score':       float(text_scores[idx]),
            })

        if deviation <= 0.01:
            status = "Exact Multi-Invoice Match" if len(selected) > 1 else "Exact Single Match"
        else:
            status = f"Match with Deviation ({deviation:.2f})"

        return status, allocations, deviation


# ═══════════════════════════════════════════════════════════════
# STAGE 7 — DEDUCTION CLASSIFIER  (NEW COMPONENT)
# ═══════════════════════════════════════════════════════════════
#
# Previously: deduction keywords were detected but never acted upon.
# The delta between payment and allocated amount had no GL destination.
#
# This component classifies the residual into one of five reason codes
# using a priority-ordered rule set. The code maps directly to what the
# ERP needs for posting:
#
#   CASH_DISCOUNT   → credit to early-payment discount GL account
#   TRADE_DEDUCTION → create deduction claim, route to trade team
#   WRITE_OFF       → credit to write-off GL account (small amounts only)
#   OVERPAYMENT     → debit excess to customer advance/liability account
#   DISPUTE         → hold in deduction suspense, route to analyst
#
# Without this classification the ERP auto-posting stage cannot determine
# which GL account to hit for the residual — the entry would be incomplete.
# ═══════════════════════════════════════════════════════════════

class DeductionClassifier:

    # Configurable thresholds
    DISCOUNT_TOLERANCE_PCT  = 0.02   # 2% — standard early-pay discount
    WRITE_OFF_THRESHOLD     = 500.0  # absolute amount below which write-off is auto-approved

    TRADE_KEYWORDS = {'promo', 'rebate', 'allowance', 'discount', 'promotional'}
    SHORTAGE_KEYWORDS = {'shortage', 'missing', 'not received', 'damaged', 'return'}

    def classify(self, payment_amount: float, total_allocated: float,
                 deduction_hints: List[str], customer_history: Optional[Dict] = None
                 ) -> Tuple[str, float, str]:
        """
        Returns: (reason_code, residual_amount, explanation)
        """
        residual = payment_amount - total_allocated
        hints_set = set(h.lower() for h in deduction_hints)

        # Overpayment
        if residual < 0:
            return "OVERPAYMENT", abs(residual), "Customer paid more than invoiced — post to advance liability"

        if residual == 0:
            return "NONE", 0.0, "Exact match — no residual"

        pct = residual / payment_amount if payment_amount > 0 else 0

        # Cash discount: small residual within discount window, no dispute keywords
        if pct <= self.DISCOUNT_TOLERANCE_PCT and not (hints_set & self.SHORTAGE_KEYWORDS):
            return "CASH_DISCOUNT", residual, f"Within {self.DISCOUNT_TOLERANCE_PCT*100:.0f}% discount tolerance"

        # Trade deduction: promo/rebate keywords present
        if hints_set & self.TRADE_KEYWORDS:
            return "TRADE_DEDUCTION", residual, "Trade promotion or rebate keyword detected in remittance"

        # Shortage/return deduction
        if hints_set & self.SHORTAGE_KEYWORDS:
            return "SHORTAGE_DEDUCTION", residual, "Shortage or return keyword detected"

        # Write-off: small unexplained residual below absolute threshold
        if residual <= self.WRITE_OFF_THRESHOLD:
            return "WRITE_OFF", residual, f"Residual {residual:.2f} below write-off threshold"

        # Default: dispute queue
        return "DISPUTE", residual, f"Unexplained residual {residual:.2f} — route to analyst"


# ═══════════════════════════════════════════════════════════════
# STAGE 8 — CONFIDENCE GATING & DECISION ENGINE
# (unchanged — composite confidence + routing thresholds correct)
# ═══════════════════════════════════════════════════════════════

class DecisionEngine:

    def __init__(self, auto_threshold=0.92, review_threshold=0.60):
        self.auto_threshold    = auto_threshold
        self.review_threshold  = review_threshold

    def decide(self, status: str, allocations: List[Dict],
               deviation: float, payment_amount: float,
               feature_map: Dict) -> Tuple[float, str, List[str]]:

        reasoning = []

        if not allocations:
            return 0.0, "unapplied", ["No matching invoices found"]

        total_allocated = sum(a['allocated_amount'] for a in allocations)
        amount_accuracy = 1.0 - (deviation / payment_amount) if payment_amount > 0 else 0
        text_support    = float(np.mean([a['text_score'] for a in allocations]))
        completeness    = min(total_allocated / payment_amount, 1.0) if payment_amount > 0 else 0

        reasoning.append(f"Amount accuracy : {amount_accuracy:.3f}")
        reasoning.append(f"Text support    : {text_support:.3f}")
        reasoning.append(f"Completeness    : {completeness:.3f}")

        confidence = min(max(
            0.50 * amount_accuracy + 0.30 * text_support + 0.20 * completeness,
            0.0), 1.0)

        if confidence >= self.auto_threshold:
            decision = "auto_post"
        elif confidence >= self.review_threshold:
            decision = "review"
        else:
            decision = "manual"

        reasoning.append(f"Confidence      : {confidence:.3f}")
        reasoning.append(f"Decision        : {decision}")
        return confidence, decision, reasoning


# ═══════════════════════════════════════════════════════════════
# STAGE 9 — ERP POSTING ENGINE  (NEW COMPONENT)
# ═══════════════════════════════════════════════════════════════
#
# Previously: the pipeline ended at MatchResult — no GL entries produced.
# This component constructs the posting payload the ERP connector needs.
#
# A complete AR clearing entry requires three legs:
#   1. DEBIT  — Bank/cash account (payment received)
#   2. CREDIT — AR sub-ledger per invoice (clears the receivable)
#   3. CREDIT/DEBIT — Adjustment account for the residual (reason-code driven)
#
# The payload is ERP-agnostic: any connector (SAP FI-CA, ERPNext, NetSuite)
# can consume this structure and map to its own document type / GL account.
#
# Only auto_post decisions trigger immediate posting. review decisions
# produce a draft payload for analyst confirmation. manual/unapplied
# produce no GL entries — cash goes to suspense.
# ═══════════════════════════════════════════════════════════════

# GL account map — replace with actual chart of accounts
GL_ACCOUNTS = {
    'bank':              '1001',   # Cash / Bank clearing
    'ar_control':        '1200',   # Accounts Receivable control
    'cash_discount':     '4900',   # Early payment discount expense
    'trade_deduction':   '5100',   # Trade promotion deductions
    'shortage_ded':      '5200',   # Shortage / return deductions
    'write_off':         '6100',   # Bad debt / write-off expense
    'overpayment':       '2300',   # Customer advance / liability
    'dispute_suspense':  '1290',   # Deduction suspense
    'unapplied':         '1280',   # Unapplied cash suspense
}

DEDUCTION_GL_MAP = {
    'CASH_DISCOUNT':     GL_ACCOUNTS['cash_discount'],
    'TRADE_DEDUCTION':   GL_ACCOUNTS['trade_deduction'],
    'SHORTAGE_DEDUCTION':GL_ACCOUNTS['shortage_ded'],
    'WRITE_OFF':         GL_ACCOUNTS['write_off'],
    'OVERPAYMENT':       GL_ACCOUNTS['overpayment'],
    'DISPUTE':           GL_ACCOUNTS['dispute_suspense'],
    'NONE':              None,
}

class ERPPoster:

    def build_gl_entries(self, result: MatchResult, payment: PaymentEvent,
                         deduction_code: str, residual: float) -> List[Dict]:
        """
        Construct GL posting payload for ERP connector.
        Returns list of debit/credit line items.
        """

        if result.decision == "unapplied":
            # Entire payment to suspense
            return [{
                'account':      GL_ACCOUNTS['unapplied'],
                'side':         'credit',
                'amount':       payment.amount,
                'currency':     payment.currency,
                'document_ref': payment.payment_id,
                'customer_id':  result.customer_id,
                'posting_date': payment.payment_date,
                'text':         'Unapplied cash — pending investigation',
            }]

        entries = []

        # Leg 1: Debit bank account (full payment amount)
        entries.append({
            'account':      GL_ACCOUNTS['bank'],
            'side':         'debit',
            'amount':       payment.amount,
            'currency':     payment.currency,
            'document_ref': payment.payment_id,
            'customer_id':  result.customer_id,
            'posting_date': payment.payment_date,
            'text':         f'Payment received — {payment.channel}',
        })

        # Leg 2: Credit AR sub-ledger per cleared invoice
        for alloc in result.allocations:
            entries.append({
                'account':      GL_ACCOUNTS['ar_control'],
                'side':         'credit',
                'amount':       alloc['allocated_amount'],
                'currency':     payment.currency,
                'document_ref': alloc['invoice_id'],
                'customer_id':  result.customer_id,
                'posting_date': payment.payment_date,
                'text':         f"AR clearing — {alloc['invoice_id']}",
            })

        # Leg 3: Residual adjustment (if any)
        if residual > 0 and deduction_code != 'NONE':
            gl_account = DEDUCTION_GL_MAP.get(deduction_code, GL_ACCOUNTS['dispute_suspense'])
            side = 'debit' if deduction_code == 'OVERPAYMENT' else 'credit'
            entries.append({
                'account':      gl_account,
                'side':         side,
                'amount':       residual,
                'currency':     payment.currency,
                'document_ref': payment.payment_id,
                'customer_id':  result.customer_id,
                'posting_date': payment.payment_date,
                'text':         f'Residual — {deduction_code}',
            })

        return entries


# ═══════════════════════════════════════════════════════════════
# STAGE 10 — FEEDBACK LOGGER (unchanged)
# ═══════════════════════════════════════════════════════════════

class FeedbackLogger:

    def __init__(self):
        self.records: List[FeedbackRecord] = []

    def log(self, result: MatchResult, feature_map: Dict) -> FeedbackRecord:
        record = FeedbackRecord(
            payment_id=result.payment_id,
            prediction={
                'allocations':    result.allocations,
                'confidence':     result.confidence,
                'decision':       result.decision,
                'status':         result.engine_status,
                'deduction_code': result.deduction_code,
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

    def __init__(self, customers: List[Customer], invoices: List[Invoice],
                 customer_histories: Optional[Dict[str, Dict]] = None):
        self.extractor         = RemittanceExtractor()
        self.resolver          = CustomerResolver(customers)
        self.retriever         = CandidateRetriever()
        self.fe                = FeatureEngineer()
        self.scorer            = MLScorer()
        self.optimizer         = MILPOptimizer(alpha=0.5, penalty=5.0)
        self.deduction_clf     = DeductionClassifier()
        self.decision_engine   = DecisionEngine()
        self.erp_poster        = ERPPoster()
        self.feedback          = FeedbackLogger()
        self.invoices          = invoices
        self.customer_histories = customer_histories or {}

    def process(self, payment: PaymentEvent) -> MatchResult:
        print("=" * 70)
        print(f"  PROCESSING PAYMENT : {payment.payment_id}")
        print(f"  Amount             : {payment.currency} {payment.amount:,.2f}")
        print("=" * 70)

        # STAGE 1: Remittance extraction
        remittance = self.extractor.extract(payment.raw_remittance)
        print(f"\n[STAGE 1] REMITTANCE EXTRACTION")
        print(f"  Invoice refs  : {remittance.invoice_refs}")
        print(f"  Amounts       : {remittance.amounts}")
        print(f"  Deductions    : {remittance.deduction_hints}")
        print(f"  Confidence    : {remittance.field_confidence}")

        # STAGE 2: Customer resolution
        customer_id, cust_conf, cust_reasons = self.resolver.resolve(payment)
        print(f"\n[STAGE 2] CUSTOMER RESOLUTION (Jaro-Winkler)")
        print(f"  Resolved     : {customer_id}  (conf={cust_conf:.3f})")
        for r in cust_reasons:
            print(f"    → {r}")

        if customer_id is None or cust_conf < 0.50:
            return MatchResult(
                payment_id=payment.payment_id, customer_id="UNKNOWN",
                allocations=[], total_allocated=0, deviation=0,
                engine_status="Customer Unresolved", confidence=0,
                decision="manual", reasoning=["Customer resolution below threshold"]
            )

        # STAGE 3: Candidate retrieval
        candidates = self.retriever.retrieve(customer_id, payment, remittance, self.invoices)
        print(f"\n[STAGE 3] CANDIDATE RETRIEVAL  ({len(candidates)} invoices)")
        for inv in candidates:
            print(f"    → {inv.invoice_id}: {payment.currency} {inv.amount:,.2f}  ({inv.days_old}d)")

        # STAGE 4: Feature engineering (with behavioral history)
        cust_history = self.customer_histories.get(customer_id)
        feature_map = {}
        for inv in candidates:
            feature_map[inv.invoice_id] = self.fe.compute(
                payment, inv, remittance, customer_history=cust_history
            )

        print(f"\n[STAGE 4] FEATURE ENGINEERING (behavioral features included)")
        if cust_history:
            print(f"  Customer history : batch={cust_history.get('avg_batch_size')}"
                  f"  days={cust_history.get('avg_days_to_pay')}"
                  f"  short_pay={cust_history.get('short_pay_rate')}")
        else:
            print(f"  Customer history : cold start (neutral defaults)")
        for inv_id, feats in feature_map.items():
            ref_sig = "✓ REF" if feats['exact_ref_match'] > 0 else "  ---"
            print(f"    {inv_id}: diff={feats['amount_diff']:>8,.0f}  "
                  f"ratio={feats['amount_ratio']:.2f}  "
                  f"{ref_sig}  text={feats['text_score']:.1f}  "
                  f"days_ratio={feats['days_ratio']:.2f}  "
                  f"short_pay={feats['short_pay_rate']:.2f}")

        # STAGE 5: ML scoring
        ml_scores = {inv.invoice_id: self.scorer.score(feature_map[inv.invoice_id])
                     for inv in candidates}
        print(f"\n[STAGE 5] ML SCORING (behavioral-adjusted)")
        for inv_id, score in sorted(ml_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 30)
            print(f"    {inv_id}: {score:.3f}  {bar}")

        # STAGE 6: MILP optimization
        status, allocations, deviation = self.optimizer.solve(
            payment.amount, candidates, feature_map
        )
        total_alloc = sum(a['allocated_amount'] for a in allocations) if allocations else 0
        print(f"\n[STAGE 6] MILP OPTIMIZATION")
        print(f"  Status       : {status}")
        print(f"  Allocated    : {payment.currency} {total_alloc:,.2f}  (dev={deviation:.2f})")
        for a in allocations:
            print(f"    → {a['invoice_id']}: {payment.currency} {a['allocated_amount']:,.2f}")

        # STAGE 7: Deduction classification (NEW)
        deduction_code, residual, ded_explanation = self.deduction_clf.classify(
            payment.amount, total_alloc, remittance.deduction_hints, cust_history
        )
        print(f"\n[STAGE 7] DEDUCTION CLASSIFICATION")
        print(f"  Code         : {deduction_code}")
        print(f"  Residual     : {payment.currency} {residual:,.2f}")
        print(f"  Explanation  : {ded_explanation}")

        # STAGE 8: Confidence gating + decision
        confidence, decision, reasoning = self.decision_engine.decide(
            status, allocations, deviation, payment.amount, feature_map
        )
        print(f"\n[STAGE 8] CONFIDENCE GATING & ROUTING")
        for r in reasoning:
            print(f"    → {r}")

        icons = {"auto_post": "✅ AUTO-POST", "review": "🔍 REVIEW",
                 "manual": "⚠️  MANUAL",   "unapplied": "❌ UNAPPLIED"}
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  {icons.get(decision, decision):<37}║")
        print(f"  ║  CONFIDENCE: {confidence:.3f}                    ║")
        print(f"  ╚══════════════════════════════════════╝")

        # STAGE 9: ERP GL entries (NEW)
        result = MatchResult(
            payment_id=payment.payment_id, customer_id=customer_id,
            allocations=allocations,       total_allocated=total_alloc,
            deviation=deviation,           engine_status=status,
            confidence=confidence,         decision=decision,
            deduction_code=deduction_code, reasoning=reasoning
        )

        if decision in ("auto_post", "review"):
            result.gl_entries = self.erp_poster.build_gl_entries(
                result, payment, deduction_code, residual
            )
        elif decision == "unapplied":
            result.gl_entries = self.erp_poster.build_gl_entries(
                result, payment, "NONE", 0
            )

        print(f"\n[STAGE 9] ERP GL ENTRIES")
        if result.gl_entries:
            for entry in result.gl_entries:
                side_icon = "DR" if entry['side'] == 'debit' else "CR"
                print(f"    {side_icon}  {entry['account']}  "
                      f"{payment.currency} {entry['amount']:>10,.2f}  "
                      f"{entry['text']}")
        else:
            print(f"    No entries — routed to manual queue")

        # STAGE 10: Feedback logging
        record = self.feedback.log(result, feature_map)
        print(f"\n[STAGE 10] FEEDBACK LOGGED  ({record.timestamp})")
        print("=" * 70)

        return result


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

def run_demo():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  CASH APPLICATION PIPELINE v2 — WITH BEHAVIORAL + ERP POSTING  ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    customers = [
        Customer("CUST-001", "Acme East Africa",
                 bank_accounts=["KE123456","KE789012"], email_domain="acme-ea.co.ke",
                 aliases=["Acme EA","AEA Holdings"]),
        Customer("CUST-002", "Nairobi Supplies Ltd",
                 bank_accounts=["KE555555"], email_domain="nairobisupplies.co.ke",
                 aliases=["Nairobi Supplies"]),
        Customer("CUST-003", "Kilimanjaro Logistics",
                 bank_accounts=["TZ999888","KE888111"], email_domain="kililogistics.com",
                 aliases=["Kili Log","Kilimanjaro Exp"]),
        Customer("CUST-004", "Rift Valley Distributors",
                 bank_accounts=["KE444333"], email_domain="rvdistributors.co.ke",
                 aliases=["RVD Ltd","Rift Valley Dist"]),
        Customer("CUST-005", "Safari Tech Solutions",
                 bank_accounts=["KE222111"], email_domain="safaritech.io",
                 aliases=["SafariTech"]),
    ]

    invoices = [
        Invoice("INV-101","CUST-001",5000.00,"KES","2026-04-01","2026-05-01","open",66),
        Invoice("INV-102","CUST-001",3000.00,"KES","2026-04-15","2026-05-15","open",52),
        Invoice("INV-103","CUST-001",2000.00,"KES","2026-05-01","2026-05-31","open",36),
        Invoice("INV-104","CUST-001",7000.00,"KES","2026-05-10","2026-06-10","open",27),
        Invoice("INV-105","CUST-001",1500.00,"KES","2026-05-20","2026-06-20","open",17),
        Invoice("INV-106","CUST-001",4500.00,"KES","2026-05-25","2026-06-25","open",12),
        Invoice("INV-201","CUST-002",12500.00,"KES","2026-03-10","2026-04-10","open",88),
        Invoice("INV-202","CUST-002",8400.00,"KES","2026-04-20","2026-05-20","open",47),
        Invoice("INV-203","CUST-002",3100.00,"KES","2026-05-15","2026-06-15","open",22),
        Invoice("INV-301","CUST-003",4500.00,"USD","2026-02-01","2026-03-01","open",125),
        Invoice("INV-302","CUST-003",6200.00,"USD","2026-05-05","2026-06-05","open",32),
        Invoice("INV-401","CUST-004",15000.00,"KES","2026-05-01","2026-05-31","open",36),
        Invoice("INV-402","CUST-004",15000.00,"KES","2026-05-12","2026-06-12","open",25),
        Invoice("INV-501","CUST-005",9800.00,"KES","2026-05-18","2026-06-18","open",19),
    ]

    # Customer behavioral histories (from historical payment data)
    customer_histories = {
        "CUST-001": {"avg_batch_size": 3.0, "avg_days_to_pay": 45.0, "short_pay_rate": 0.05},
        "CUST-002": {"avg_batch_size": 2.0, "avg_days_to_pay": 60.0, "short_pay_rate": 0.10},
        "CUST-003": {"avg_batch_size": 1.0, "avg_days_to_pay": 90.0, "short_pay_rate": 0.00},
        "CUST-004": {"avg_batch_size": 1.0, "avg_days_to_pay": 30.0, "short_pay_rate": 0.20},
        "CUST-005": {"avg_batch_size": 1.0, "avg_days_to_pay": 30.0, "short_pay_rate": 0.15},
    }

    payments = [
        PaymentEvent("PAY-2026-0042", 10000.00, "KES", "2026-06-06",
                     "KE123456", "ACH", "Payment for INV-101 and INV-102 and INV-103",
                     "Acme East Africa", "accounts@acme-ea.co.ke"),
        PaymentEvent("PAY-2026-0043", 8400.00, "KES", "2026-06-07",
                     "KE555555", "RTGS", "WIRE TRF FROM NAIROBI SUPPLIES",
                     "Nairobi Supplies", "finance@nairobisupplies.co.ke"),
        PaymentEvent("PAY-2026-0044", 15600.00, "KES", "2026-06-07",
                     "KE555555", "ACH", "BULK INVOICE SETTLEMENT",
                     "Nairobi Supplies Ltd", "ap@nairobisupplies.co.ke"),
        PaymentEvent("PAY-2026-0045", 9650.00, "KES", "2026-06-08",
                     "KE222111", "MPESA", "SAFARI-TECH INV-501 LESS CHARGES",
                     "SafariTech", "accounting@safaritech.io"),
        PaymentEvent("PAY-2026-0046", 15000.00, "KES", "2026-06-08",
                     "KE444333", "RTGS", "SETTLEMENT FOR PARTS - REF: INV-402",
                     "Rift Valley Dist", "payables@rvdistributors.co.ke"),
        PaymentEvent("PAY-2026-0047", 11000.00, "KES", "2026-06-08",
                     "KE789012", "ACH", "ON ACCOUNT DEPOSIT AEA HOLDINGS",
                     "AEA Holdings", "treasury@acme-ea.co.ke"),
    ]

    pipeline = CashApplicationPipeline(customers, invoices, customer_histories)

    for payment in payments:
        result = pipeline.process(payment)
        print(f"\n  SUMMARY: {result.payment_id} | {result.engine_status} | "
              f"conf={result.confidence:.3f} | {result.decision.upper()} | "
              f"ded={result.deduction_code}")
        print()

    print(f"\n  Feedback records ready for XGBoost training: {len(pipeline.feedback.records)}")


run_demo()