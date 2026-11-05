/**
 * Redaction profiles.
 *
 * A fine-tuning dataset is the one artifact in this pipeline that outlives the
 * session it came from: it gets shared, uploaded to a remote trainer, and baked
 * into weights that cannot be un-remembered. So redaction is a default-on stage
 * with an explicit profile per data source, not a checkbox someone remembers.
 *
 * Profiles are ordered by strictness:
 *   key   — API keys, tokens, private keys, connection strings (always on)
 *   pii   — emails, phone numbers, ID-card shapes (default on)
 *   path  — absolute home paths and user names in commands (default on)
 *   full  — everything above plus free-form numbers and URLs (off; destroys
 *           the very content most samples are about)
 *
 * Each rule reports *what* it redacted and how many times, never the matched
 * value itself — the audit trail must be safe to paste into an issue.
 */

/** Luhn check for card-shaped digit runs; ordinary identifiers must not trip the rule. */
function luhnOk(digits) {
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i -= 1) {
    let d = digits.charCodeAt(i) - 48;
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

const RULES = [
  {
    id: 'private-key',
    tier: 'key',
    label: 'PEM private key block',
    pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
    replace: '[REDACTED_PRIVATE_KEY]',
  },
  {
    id: 'sk-key',
    tier: 'key',
    label: 'provider API key (sk-/pk-/ghp_/xox-/AKIA...)',
    pattern: /\b(sk-[A-Za-z0-9_-]{16,}|pk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})\b/g,
    replace: '[REDACTED_KEY]',
  },
  {
    id: 'bearer',
    tier: 'key',
    label: 'Authorization header / Bearer token',
    pattern: /\b(Authorization\s*:\s*(?:Bearer|Basic|Token)\s+)[A-Za-z0-9._~+/=-]{8,}/gi,
    replace: '$1[REDACTED]',
  },
  {
    id: 'kv-secret',
    tier: 'key',
    label: 'password/apiKey/token assignment',
    pattern: /\b((?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*["']?)[^\s"',;]{6,}/gi,
    replace: '$1[REDACTED]',
  },
  {
    id: 'dsn',
    tier: 'key',
    label: 'connection string with inline credentials',
    pattern: /\b((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|mssql|ftp):\/\/[^:@\s/]+:)[^@\s/]+@/gi,
    replace: '$1[REDACTED]@',
  },
  {
    id: 'email',
    tier: 'pii',
    label: 'email address',
    pattern: /\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b/g,
    replace: '[REDACTED_EMAIL]',
  },
  {
    id: 'cn-id',
    tier: 'pii',
    label: 'Chinese resident ID number',
    // 17 digits + 0-9/X, but only where it is plausibly an ID: not glued to a
    // longer digit run (which would be a timestamp or a hash), and a valid
    // birth-year prefix. The check digit is verified in `verify` below.
    pattern: /(?<![\dA-Za-z])\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![\dA-Za-z])/g,
    replace: '[REDACTED_ID]',
    accept: (match) => cnIdCheckDigitOk(match),
  },
  {
    id: 'phone-cn',
    tier: 'pii',
    label: 'Chinese mobile number',
    pattern: /(?<![\d])(?:\+?86[-\s]?)?1[3-9]\d{9}(?![\d])/g,
    replace: '[REDACTED_PHONE]',
  },
  {
    id: 'card',
    tier: 'pii',
    label: 'payment card number',
    pattern: /(?<![\d])(?:\d[ -]?){13,18}\d(?![\d])/g,
    replace: '[REDACTED_CARD]',
    accept: (match) => {
      const digits = match.replace(/\D/g, '');
      return digits.length >= 13 && digits.length <= 19 && luhnOk(digits);
    },
  },
  {
    id: 'home-path-win',
    tier: 'path',
    label: 'Windows user profile path',
    // Case-insensitive on the drive so `c:\Users\...` is caught too — the
    // captured user name contains no regex metacharacters (it is `[^\\/]+`).
    pattern: /\b([A-Za-z]:[\\/]+Users[\\/]+)([^\\/:*?"<>|\r\n]+)/gi,
    replace: (match, prefix) => `${prefix}[USER]`,
  },
  {
    id: 'home-path-posix',
    tier: 'path',
    label: 'POSIX home path',
    pattern: /(\/(?:home|Users)\/)([A-Za-z0-9._-]+)/g,
    replace: (match, prefix) => `${prefix}[USER]`,
  },
  {
    id: 'url',
    tier: 'full',
    label: 'URL (full profile only)',
    pattern: /\bhttps?:\/\/[^\s"'<>)]+/g,
    replace: '[REDACTED_URL]',
  },
];

/** Chinese ID numbers carry a MOD 11-2 check digit; a random 18-digit run fails it. */
function cnIdCheckDigitOk(id) {
  const weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
  const checks = '10X98765432';
  let sum = 0;
  for (let i = 0; i < 17; i += 1) sum += (id.charCodeAt(i) - 48) * weights[i];
  return checks[sum % 11] === id.charAt(17).toUpperCase();
}

export const REDACTION_PROFILES = {
  off: [],
  key: ['key'],
  pii: ['key', 'pii'],
  path: ['key', 'pii', 'path'],
  full: ['key', 'pii', 'path', 'full'],
};

export const DEFAULT_REDACTION = 'path';

/**
 * Apply a profile to one string.
 *
 * @returns {{ text: string, hits: Array<{ rule: string, label: string, count: number }> }}
 */
export function redactText(text, profile = DEFAULT_REDACTION) {
  const tiers = REDACTION_PROFILES[profile];
  if (tiers === undefined) throw new Error(`unknown redaction profile: ${profile}`);
  if (typeof text !== 'string' || text === '') return { text: text ?? '', hits: [] };
  let out = text;
  const hits = [];
  for (const rule of RULES) {
    if (!tiers.includes(rule.tier)) continue;
    let count = 0;
    out = out.replace(rule.pattern, (...args) => {
      if (typeof rule.accept === 'function' && !rule.accept(args[0])) return args[0];
      count += 1;
      return typeof rule.replace === 'function' ? rule.replace(...args) : rule.replace;
    });
    if (count > 0) hits.push({ rule: rule.id, label: rule.label, count });
  }
  return { text: out, hits };
}

/** Deep redaction of any JSON-ish value (messages carry nested content blocks). */
export function redactDeep(value, profile = DEFAULT_REDACTION) {
  const hits = [];
  const walk = (node) => {
    if (typeof node === 'string') {
      const result = redactText(node, profile);
      hits.push(...result.hits);
      return result.text;
    }
    if (Array.isArray(node)) return node.map(walk);
    if (node !== null && typeof node === 'object') {
      const out = {};
      for (const [key, child] of Object.entries(node)) out[key] = walk(child);
      return out;
    }
    return node;
  };
  return { value: walk(value), hits };
}
