import math
import random
from collections import Counter, defaultdict


class Ngram:
    def __init__(self, cfg):
        assert cfg.n >= 2, "N must be at least 2 for N-gram models."
        self.cfg = cfg
        self.cnt1 = defaultdict(Counter)    # how many times each (context, next) pair occurs
        self.cnt2 = Counter()               # how many times each context occurs
        self.vocab = {'<BOS>', '<EOS>'}
        print(f"Ngram model initialized with N={cfg.n} and k={cfg.k}")

    def train(self, sequences):
        """
        sequences: iterable of token sequences (each sequence is a list/sequence of str).
        Side effects: updates self.cnt1 (defaultdict(Counter)), self.cnt2 (Counter), and self.vocab (set).
        """
        print(f"Starting training for N={self.cfg.n} model...")
        n = self.cfg.n

        cnt1 = self.cnt1        # Counter of (context, next) pairs
        cnt2 = self.cnt2        # Counter of contexts
        vocab = self.vocab

        for seq in sequences:
            padded = ['<BOS>'] * (n - 1) + list(seq) + ['<EOS>']
            last_idx = len(padded) - (n - 1)
            for i in range(last_idx):
                context = tuple(padded[i : i + n - 1])
                target = padded[i + n - 1]
                cnt1[context][target] += 1
                cnt2[context] += 1
                vocab.add(target)

    def prob(self, context):
        """
        Given a context, returns a probability distribution over the next token using add-k smoothing
        P(w | c) = (count(c, w) + k) / (count(c) + k * |V|)
        """
        n = self.cfg.n
        k = self.cfg.k
        V = len(self.vocab)
        
        ctx_list = list(context)[-(n-1):]
        if len(ctx_list) < n-1:
            ctx_list = ['<BOS>']*(n-1-len(ctx_list)) + ctx_list
        
        ctx_tuple = tuple(ctx_list)
        A = self.cnt1.get(ctx_tuple, Counter())
        B = self.cnt2.get(ctx_tuple, 0) + k * V
        
        if B <= 0:
            B = k * V if V > 0 else 1.0 # Avoid division by zero
            
        return {w: (A.get(w, 0) + k) / B for w in self.vocab}

    def topk(self, context, topk=10):
        """Returns the top-k most probable next tokens."""
        dist = self.prob(context)
        return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:topk]

    def generate_next(self, context):
        dist = self.prob(context)
        items = list(dist.items())
        tokens, probs = zip(*items)
        
        # Using random.choices is simpler and more robust
        try:
            return random.choices(tokens, weights=probs, k=1)[0]
        except ValueError: # handles case where all probs are zero
            return '<EOS>'

    def generate(self, context, max_len=50):
        out = []
        ctx = list(context)
        for _ in range(max_len):
            t = self.generate_next(ctx)
            if t == '<EOS>':
                break
            if t == '}':
                out.append(t)
                break
            out.append(t)
            ctx.append(t)
        return out

    def log_prob(self, seq):
        """
        Prevents underflow
        """
        n = self.cfg.n
        s = ['<BOS>']*(n-1) + seq + ['<EOS>']
        log_p = 0.0
        for i in range(len(s) - n + 1):
            ctx = s[i:i+n-1]
            nxt = s[i+n-1]
            p = self.prob(ctx).get(nxt, 1e-12) # Use a small floor probability
            log_p += math.log(p if p > 0 else 1e-12)
        return log_p

    def PPL(self, seqs):
        """
        PPL = exp(- (1/T) * sum(log P(w_i | context)))
        """
        T = 0       # total number of tokens (including EOS)
        P = 0.0     # cumulative probability
        for seq in seqs:
            T += len(seq) + 1  # +1 for EOS
            P += self.log_prob(seq)
        return float('inf') if T == 0 else (math.exp(-P / T))
