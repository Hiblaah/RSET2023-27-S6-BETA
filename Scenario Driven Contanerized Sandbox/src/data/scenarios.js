export const scenarios = [
  {
    id: 'ecommerce-tax',
    category: 'backend',
    title: 'E-Commerce Tax Engine',
    description: 'Fix the tax calculation logic in a multi-file shopping cart system. The current implementation only returns the subtotal.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'index.js': {
        name: 'index.js',
        language: 'javascript',
        content: `import { ShoppingCart } from './cart.js';
const cart = new ShoppingCart(0.1); // 10% tax
cart.addItem({ price: 100, quantity: 1 });
const total = cart.calculateTotal();
console.log('RESULT_TOTAL:' + total);`
      },
      'cart.js': {
        name: 'cart.js',
        language: 'javascript',
        content: `export class ShoppingCart {
  constructor(taxRate = 0.08) {
    this.items = [];
    this.taxRate = taxRate;
  }

  addItem(item) {
    this.items.push(item);
  }

  // BUG: This only returns subtotal, ignoring taxRate
  calculateTotal() {
    const subtotal = this.items.reduce((sum, item) => {
      return sum + (item.price * item.quantity);
    }, 0);
    
    return subtotal;
  }
}`
      },
      'constants.js': {
        name: 'constants.js',
        language: 'javascript',
        content: `export const DEFAULT_TAX_RATE = 0.08;\nexport const MAX_ITEMS = 100;`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Tax Calculation Scenario\n\nUpdate `cart.js` to include the tax rate in the final total. Formula: subtotal * (1 + taxRate).'
      }
    },
    constraints: ['Final total must include tax.', 'Ensure subtotal is calculated first.'],
    testCheck: `
            const cartCode = files['cart.js']?.content || '';
            const hasFormula = cartCode.includes('taxRate') && (cartCode.includes('*') || cartCode.includes('+'));
            const runtimePassed = stdout.includes('RESULT_TOTAL:110'); // Accept 110 or 110.000...
            return hasFormula && runtimePassed;
        `
  },
  {
    id: 'auth-session',
    category: 'backend',
    title: 'Secure Session Management',
    description: 'Build a session validation layer using separate logic and constant modules. Prevent unauthorized access for expired tokens.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'index.js': {
        name: 'index.js',
        language: 'javascript',
        content: `import { isSessionActive } from './sessionManager.js';
import { SESSION_TIMEOUT } from './config.js';

const now = Date.now();
const validSession = { createdAt: now - 1000 };
const expiredSession = { createdAt: now - SESSION_TIMEOUT - 1000 };

console.log('TEST_VALID:' + isSessionActive(validSession));
console.log('TEST_EXPIRED:' + isSessionActive(expiredSession));`
      },
      'sessionManager.js': {
        name: 'sessionManager.js',
        language: 'javascript',
        content: `import { SESSION_TIMEOUT } from './config.js';\n\nexport function isSessionActive(session) {\n  if (!session) return false;\n  // TODO: Check if session.createdAt + SESSION_TIMEOUT > Date.now()\n  return true;\n}`
      },
      'config.js': {
        name: 'config.js',
        language: 'javascript',
        content: `export const SESSION_TIMEOUT = 3600000; // 1 hour in ms`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Session Security\n\nImplement the expiration check in `sessionManager.js` using the timeout from `config.js`.'
      }
    },
    constraints: ['Verify session existence.', 'Check timestamp against current time.'],
    testCheck: `
            const sessionMgrCode = files['sessionManager.js']?.content || '';
            const hasLogic = sessionMgrCode.includes('SESSION_TIMEOUT') && sessionMgrCode.includes('Date.now()');
            const noTodo = !sessionMgrCode.includes('TODO');
            const runtimePassed = stdout.includes('TEST_VALID:true') && stdout.includes('TEST_EXPIRED:false');
            return hasLogic && noTodo && runtimePassed;
        `
  },
  {
    id: 'inventory-alerts',
    category: 'backend',
    title: 'Intelligent Stock Alerts',
    description: 'Refactor the notification system to use a data-driven threshold model instead of hardcoded values.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'notifier.js': {
        name: 'notifier.js',
        language: 'javascript',
        content: `import { THRESHOLDS } from './data.js';\n\nexport function shouldAlert(itemType, quantity) {\n  // TODO: Get threshold from THRESHOLDS map and compare\n  return false;\n}`
      },
      'data.js': {
        name: 'data.js',
        language: 'javascript',
        content: `export const THRESHOLDS = {\n  electronics: 5,\n  clothing: 10,\n  food: 20\n};`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Inventory Alerts\n\nConnect the notifier to the data module to check specific category thresholds.'
      }
    },
    constraints: ['Use THRESHOLDS object.', 'Return true if quantity <= threshold.'],
    testCheck: (code) => code.includes('THRESHOLDS[') || code.includes('THRESHOLDS.')
  },
  {
    id: 'api-error-boundary',
    category: 'backend',
    title: 'Graceful API Degrader',
    description: 'Implement a resilience layer that falls back to localized mock data when the primary API client fails.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'apiClient.js': {
        name: 'apiClient.js',
        language: 'javascript',
        content: `import { MOCK_DATA } from './fallback.js';\n\nexport async function getSafeData(url) {\n  try {\n    const res = await fetch(url);\n    return await res.json();\n  } catch (err) {\n    // TODO: Return MOCK_DATA instead of throwing\n    throw err;\n  }\n}`
      },
      'fallback.js': {
        name: 'fallback.js',
        language: 'javascript',
        content: `export const MOCK_DATA = { status: "cached", items: [] };`
      }
    },
    constraints: ['Use try-catch block.', 'Import and return fallback data.'],
    testCheck: (code) => code.includes('catch') && code.includes('MOCK_DATA')
  },
  {
    id: 'theme-orchestrator',
    category: 'backend',
    title: 'Multi-Context Theme Manager',
    description: 'Coordinate system preferences and user overrides across a stylesheet and a logic controller.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'themeController.js': {
        name: 'themeController.js',
        language: 'javascript',
        content: `import { THEMES } from './palette.js';\n\nexport function getThemeClass(isDark) {\n  // TODO: Return THEMES.dark if isDark is true, else THEMES.light\n  return "";\n}`
      },
      'palette.js': {
        name: 'palette.js',
        language: 'javascript',
        content: `export const THEMES = {\n  dark: "bg-slate-900 text-white",\n  light: "bg-white text-slate-900"\n};`
      }
    },
    constraints: ['Use external palette mapping.', 'Support conditional toggle.'],
    testCheck: (code) => code.includes('THEMES.dark') && code.includes('THEMES.light')
  },
  {
    id: 'data-transformer-pro',
    category: 'backend',
    title: 'Schema-Driven Transformer',
    description: 'Map raw external data into a validated internal schema using shared transformation rules.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'transformer.js': {
        name: 'transformer.js',
        language: 'javascript',
        content: `import { FIELD_MAP } from './schema.js';\n\nexport function transform(rawData) {\n  return rawData.map(item => {\n    // TODO: Rename keys based on FIELD_MAP\n    return item;\n  });\n}`
      },
      'schema.js': {
        name: 'schema.js',
        language: 'javascript',
        content: `export const FIELD_MAP = {\n  external_id: "id",\n  user_name: "name",\n  email_address: "email"\n};`
      }
    },
    constraints: ['Iterate through mappings.', 'Do not mutate original objects.'],
    testCheck: (code) => code.includes('FIELD_MAP') && code.includes('.map')
  },
  {
    id: 'rbac-gatekeeper',
    category: 'backend',
    title: 'Distributed RBAC Guard',
    description: 'Check nested permissions across a multi-role architecture to protect specific system actions.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'guard.js': {
        name: 'guard.js',
        language: 'javascript',
        content: `import { PERMISSIONS } from './roles.js';\n\nexport function hasAccess(role, action) {\n  const allowed = PERMISSIONS[role] || [];\n  // TODO: Check if action exists in allowed array\n  return false;\n}`
      },
      'roles.js': {
        name: 'roles.js',
        language: 'javascript',
        content: `export const PERMISSIONS = {\n  admin: ["create", "edit", "delete", "view"],\n  editor: ["edit", "view"],\n  guest: ["view"]\n};`
      }
    },
    constraints: ['Support dynamic role lookup.', 'Handle undefined roles gracefully.'],
    testCheck: (code) => code.includes('PERMISSIONS[') && code.includes('.includes(')
  },
  {
    id: 'search-efficiency',
    category: 'backend',
    title: 'Multi-File Debouncer',
    description: 'Implement a high-performance debouncer for search triggers that interacts with an external API module.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'debounce.js': {
        name: 'debounce.js',
        language: 'javascript',
        content: `export function createDebouncer(fn, delay) {\n  let timeoutId;\n  return (...args) => {\n    // TODO: Clear previous timeout and set new one\n  };\n}`
      },
      'searchUI.js': {
        name: 'searchUI.js',
        language: 'javascript',
        content: `import { createDebouncer } from './debounce.js';\n\n// Usage example for tests\nexport const debouncedSearch = createDebouncer(() => console.log("Searching..."), 300);`
      }
    },
    constraints: ['Use closures for state.', 'Handle rapid event firing.'],
    testCheck: (code) => code.includes('clearTimeout') && code.includes('setTimeout')
  },
  {
    id: 'finance-wizard',
    category: 'backend',
    title: 'Compound Interest Studio',
    description: 'Build a financial calculation module that uses separate interest rate and compounding factor configurations.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'calc.js': {
        name: 'calc.js',
        language: 'javascript',
        content: `import { FACTORS } from './settings.js';\n\nexport function calculateGrowth(p, r, t, frequency) {\n  const n = FACTORS[frequency] || 1;\n  // Formula: A = P(1 + r/n)^(nt)\n  return p * (1 + r);\n}`
      },
      'settings.js': {
        name: 'settings.js',
        language: 'javascript',
        content: `export const FACTORS = {\n  monthly: 12,\n  quarterly: 4,\n  annually: 1\n};`
      }
    },
    constraints: ['Support multiple compounding frequencies.', 'Correct math implementation.'],
    testCheck: (code) => code.includes('Math.pow') || code.includes('**')
  },
  {
    id: 'seo-slug-generator',
    category: 'backend',
    title: 'SEO Slug Workflows',
    description: 'Cleanse and transform website titles into production-ready URLs using a regex-based cleansing module.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'slugger.js': {
        name: 'slugger.js',
        language: 'javascript',
        content: `import { CLEAN_REGEX } from './patterns.js';\n\nexport function toSlug(text) {\n  // TODO: Lowercase, replace non-alphanumeric (using CLEAN_REGEX) with '-', and trim\n  return text;\n}`
      },
      'patterns.js': {
        name: 'patterns.js',
        language: 'javascript',
        content: `export const CLEAN_REGEX = /[^a-z0-9]+/g;`
      }
    },
    constraints: ['Use regex for replacement.', 'Clean start/end dashes.'],
    testCheck: (code) => code.includes('CLEAN_REGEX') && code.includes('replace')
  },
  {
    id: 'inventory-processor',
    category: 'backend',
    title: 'Order Status Workflow',
    description: 'Manage complex order state transitions across a dedicated processor and a status mapping file.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'orderProcessor.js': {
        name: 'orderProcessor.js',
        language: 'javascript',
        content: `import { STATES } from './statusCodes.js';\n\nexport function advanceOrder(currentStatus) {\n  // TODO: Return next available state from STATES workflow\n  return currentStatus;\n}`
      },
      'statusCodes.js': {
        name: 'statusCodes.js',
        language: 'javascript',
        content: `export const STATES = [\n  "PENDING",\n  "PROCESSING",\n  "SHIPPED",\n  "DELIVERED"\n];`
      }
    },
    constraints: ['Find index of current state.', 'Handle "DELIVERED" boundary.'],
    testCheck: (code) => code.includes('STATES.indexOf') || code.includes('STATES[')
  },
  {
    id: 'file-utility-kit',
    category: 'backend',
    title: 'Readable Size Formatter',
    description: 'Coordinate byte-to-string conversions using a unit-map file to support KB, MB, and GB precision.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'formatter.js': {
        name: 'formatter.js',
        language: 'javascript',
        content: `import { UNITS } from './units.js';\n\nexport function formatSize(bytes) {\n  // TODO: Iterate units and divide by 1024 until correct tier found\n  return bytes + " " + UNITS[0];\n}`
      },
      'units.js': {
        name: 'units.js',
        language: 'javascript',
        content: `export const UNITS = ["B", "KB", "MB", "GB", "TB"];`
      }
    },
    constraints: ['Loop through units.', 'Limit to 2 decimal places.'],
    testCheck: (code) => code.includes('1024') && code.includes('UNITS[')
  },
  {
    id: 'config-merger-v2',
    category: 'backend',
    title: 'Recursive Config Union',
    description: 'Merge deep JSON configurations from multiple sources into a single unified application state.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'merger.js': {
        name: 'merger.js',
        language: 'javascript',
        content: `export function deepMerge(target, source) {\n  // TODO: Recursively merge keys if both are objects\n  return { ...target, ...source }; \n}`
      },
      'defaults.js': {
        name: 'defaults.js',
        language: 'javascript',
        content: `export const APP_DEFAULTS = {\n  api: { timeout: 1000, retry: true },\n  ui: { theme: "dark" }\n};`
      }
    },
    constraints: ['Handle nested objects.', 'Avoid overwriting entire branches.'],
    testCheck: (code) => code.includes('typeof') && code.includes('deepMerge')
  },
  {
    id: 'color-parser-studio',
    category: 'backend',
    title: 'Hex to RGB Converter',
    description: 'Implement a parsing workflow that converts HEX strings into RGB objects using bitwise or string-slice logic.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'parser.js': {
        name: 'parser.js',
        language: 'javascript',
        content: `export function hexToRgb(hex) {\n  // TODO: Extract R, G, B components from #RRGGBB\n  return { r: 0, g: 0, b: 0 };\n}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Color Logic\n\nConvert a 6-digit hex string into its numeric RGB components.'
      }
    },
    constraints: ['Handle optional # prefix.', 'Return numeric values.'],
    testCheck: (code) => code.includes('parseInt') && code.includes('16')
  },
  {
    id: 'security-analyzer',
    category: 'backend',
    title: 'Password Pulse Gauge',
    description: 'Create a security analyzer that scores passwords based on complex patterns defined in a shared policy file.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'analyzer.js': {
        name: 'analyzer.js',
        language: 'javascript',
        content: `import { POLICY } from './policy.js';\n\nexport function checkSecurity(pass) {\n  let score = 0;\n  // TODO: Check against each POLICY regex\n  return score;\n}`
      },
      'policy.js': {
        name: 'policy.js',
        language: 'javascript',
        content: `export const POLICY = [\n  { name: "length", regex: /.{8,}/ },\n  { name: "digit", regex: /[0-9]/ },\n  { name: "special", regex: /[^a-zA-Z0-9]/ }\n];`
      }
    },
    constraints: ['Iterate policy array.', 'Increment score for each match.'],
    testCheck: (code) => code.includes('POLICY.forEach') || code.includes('POLICY.reduce')
  },
  {
    id: 'data-dedupe-engine',
    category: 'backend',
    title: 'Unique Entity Resolver',
    description: 'Deduplicate complex data lists based on a unique identifier key specified in a settings module.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'deduper.js': {
        name: 'deduper.js',
        language: 'javascript',
        content: `export function removeDuplicates(list, uniqueId) {\n  // TODO: Filter list to keep only first occurrence of uniqueId\n  return list;\n}`
      },
      'testData.js': {
        name: 'testData.js',
        language: 'javascript',
        content: `export const SAMPLE_USERS = [\n  { id: 1, name: "A" },\n  { id: 1, name: "A" },\n  { id: 2, name: "B" }\n];`
      }
    },
    constraints: ['Maintain original order.', 'Efficient handling.'],
    testCheck: (code) => code.includes('Map') || code.includes('Set') || code.includes('filter')
  },
  {
    id: 'time-relative-pro',
    category: 'backend',
    title: 'Social Media Timestamps',
    description: 'Convert timestamps into "ago" strings using a multi-threshold interval mapping system.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'timeAgo.js': {
        name: 'timeAgo.js',
        language: 'javascript',
        content: `import { INTERVALS } from './thresholds.js';\n\nexport function getTimeAgo(date) {\n  const seconds = Math.floor((Date.now() - date) / 1000);\n  // TODO: Find correct interval and return it\n  return "just now";\n}`
      },
      'thresholds.js': {
        name: 'thresholds.js',
        language: 'javascript',
        content: `export const INTERVALS = [\n  { label: "year", seconds: 31536000 },\n  { label: "month", seconds: 2592000 },\n  { label: "day", seconds: 86400 },\n  { label: "hour", seconds: 3600 },\n  { label: "minute", seconds: 60 }\n];`
      }
    },
    constraints: ['Iterate through intervals.', 'Return plural variants if count > 1.'],
    testCheck: (code) => code.includes('INTERVALS.find') || code.includes('INTERVALS.forEach')
  },
  {
    id: 'form-validation-kit',
    category: 'backend',
    title: 'Cross-Field Form Validator',
    description: 'Implement a validation engine that checks correlations between multiple fields (e.g., password and confirmation).',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'formValidator.js': {
        name: 'formValidator.js',
        language: 'javascript',
        content: `export function validateFields(fields) {\n  // fields = { password, confirmPassword }\n  // TODO: Check if they match\n  return { valid: true };\n}`
      },
      'messages.js': {
        name: 'messages.js',
        language: 'javascript',
        content: `export const ERRORS = {\n  mismatch: "Passwords must match",\n  tooShort: "Too short"\n};`
      }
    },
    constraints: ['Use external error messages.', 'Check field equality.'],
    testCheck: (code) => code.includes('ERRORS.') && code.includes('===')
  },
  {
    id: 'url-query-builder',
    category: 'backend',
    title: 'Dynamic URL Assembler',
    description: 'Construct complex URL strings with query parameters using a shared base configuration module.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'assembler.js': {
        name: 'assembler.js',
        language: 'javascript',
        content: `import { BASE_URL } from './config.js';\n\nexport function buildUrl(endpoint, params) {\n  // TODO: Append params as query string to BASE_URL + endpoint\n  return BASE_URL + endpoint;\n}`
      },
      'config.js': {
        name: 'config.js',
        language: 'javascript',
        content: `export const BASE_URL = "https://api.sandbox.com/v1";`
      }
    },
    constraints: ['Handle multiple parameters.', 'Proper URL encoding.'],
    testCheck: (code) => code.includes('?') && (code.includes('URLSearchParams') || code.includes('&'))
  },
  {
    id: 'localization-manager',
    category: 'backend',
    title: 'Multi-Language Store',
    description: 'Build a translation engine that switches between language bundles based on a locale code.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'i18n.js': {
        name: 'i18n.js',
        language: 'javascript',
        content: `import { BUNDLES } from './locales.js';\n\nexport function translate(locale, key) {\n  // TODO: Return nested value from BUNDLES[locale]\n  return key;\n}`
      },
      'locales.js': {
        name: 'locales.js',
        language: 'javascript',
        content: `export const BUNDLES = {\n  en: { welcome: "Welcome" },\n  es: { welcome: "Bienvenido" }\n};`
      }
    },
    constraints: ['Support deep nesting.', 'Fallback if key missing.'],
    testCheck: (code) => code.includes('BUNDLES[locale]') || code.includes('BUNDLES.')
  },
  {
    id: 'payment-rounding-bug',
    category: 'backend',
    title: 'Payment Precision Bug (Floating Point)',
    description: 'Debug a checkout system where fractional cents are lost during calculations, causing revenue leakage. The bug occurs when multiplying decimal values.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'checkout.js': {
        name: 'checkout.js',
        language: 'javascript',
        content: `import { applyDiscount, TAX_RATE, roundPrice } from './pricing.js';

export function calculateTotal(items) {
  let subtotal = 0;
  
  for (const item of items) {
    subtotal += item.price * item.quantity;
  }
  
  const discounted = applyDiscount(subtotal, 0.15); // 15% discount
  const tax = discounted * TAX_RATE;
  const total = discounted + tax;
  
  console.log('SUBTOTAL:' + subtotal);
  console.log('DISCOUNTED:' + discounted);
  console.log('TAX:' + tax);
  console.log('TOTAL:' + total);
  console.log('FINAL:' + roundPrice(total));
  
  return total;
}`
      },
      'pricing.js': {
        name: 'pricing.js',
        language: 'javascript',
        content: `export const TAX_RATE = 0.0875; // 8.75%

export function applyDiscount(amount, discount) {
  // BUG: Direct multiplication causes floating point errors
  return amount - (amount * discount);
}

export function roundPrice(price) {
  // BUG: Not properly rounding to 2 decimal places
  return Math.round(price * 100) / 100;
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Payment Precision Bug\n\nTest: Purchase 3 items at $19.99 each with 15% discount and 8.75% tax.\nExpected FINAL: 52.47\n\nThe system is losing precision. Fix the rounding logic.'
      }
    },
    constraints: ['Use proper decimal math.', 'Round consistently to 2 places.'],
    testCheck: `
            const pricingCode = files['pricing.js']?.content || '';
            const hasRounding = pricingCode.includes('Math.round') && pricingCode.includes('100');
            const runtimePassed = stdout.includes('FINAL:52.47');
            return hasRounding && runtimePassed;
        `
  },
  {
    id: 'database-connection-leak',
    category: 'backend',
    title: 'Connection Pool Memory Leak',
    description: 'Debug a service that fails to close database connections properly. Connections pile up and exhaust the pool, causing "ECONNREFUSED" errors.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'userService.js': {
        name: 'userService.js',
        language: 'javascript',
        content: `import { getConnection } from './database.js';

export async function getUserById(id) {
  const conn = getConnection();
  
  try {
    const user = conn.query('SELECT * FROM users WHERE id = ?', [id]);
    // BUG: Connection is never closed!
    return user;
  } catch (err) {
    console.log('ERROR:Connection failed');
    return null;
  }
}

export async function getUsersWithPosts(userId) {
  const conn = getConnection();
  
  try {
    const user = conn.query('SELECT * FROM users WHERE id = ?', [userId]);
    const posts = conn.query('SELECT * FROM posts WHERE user_id = ?', [userId]);
    // BUG: Early return skips closeConnection call below
    if (!user) return null;
    
    const result = { user, posts };
    // BUG: This line never executes if user is null
    closeConnection(conn);
    return result;
  } catch (err) {
    console.log('ERROR:Query failed');
  }
}`
      },
      'database.js': {
        name: 'database.js',
        language: 'javascript',
        content: `let activeConnections = 0;
const MAX_CONNECTIONS = 5;

export function getConnection() {
  if (activeConnections >= MAX_CONNECTIONS) {
    throw new Error('ECONNREFUSED:Connection pool exhausted');
  }
  activeConnections++;
  console.log('CONN_OPEN:' + activeConnections);
  return { query: () => ({ id: 1, name: 'Test' }) };
}

export function closeConnection(conn) {
  activeConnections--;
  console.log('CONN_CLOSED:' + activeConnections);
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Connection Leak Bug\n\nFix: Ensure all connection lifetimes are properly managed.\n- Use try-finally or error handlers\n- Close connections even when errors occur\n- Never skip cleanup in early returns'
      }
    },
    constraints: ['Use try-finally.', 'Close in all code paths.'],
    testCheck: `
            const serviceCode = files['userService.js']?.content || '';
            const hasFinally = serviceCode.includes('finally');
            const hasClose = serviceCode.includes('closeConnection');
            const runtimePassed = stdout.includes('CONN_CLOSED:0');
            return hasFinally && hasClose && runtimePassed;
        `
  },
  {
    id: 'race-condition-counter',
    category: 'backend',
    title: 'Race Condition in Async Counter',
    description: 'Debug an inventory system where concurrent requests increment the counter incorrectly. Multiple async operations read-modify-write without coordination.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'inventory.js': {
        name: 'inventory.js',
        language: 'javascript',
        content: `import { INVENTORY } from './state.js';

export async function decrementStock(productId, quantity) {
  // BUG: Race condition - read then write without atomicity
  const current = INVENTORY[productId];
  
  // Simulate async operation (like DB call)
  await new Promise(resolve => setTimeout(resolve, 10));
  
  if (current < quantity) {
    console.log('FAILED:Insufficient stock');
    return false;
  }
  
  // By this time, another thread may have decremented!
  INVENTORY[productId] = current - quantity;
  console.log('SUCCESS:' + INVENTORY[productId]);
  return true;
}

export function addStock(productId, quantity) {
  // BUG: Same issue here
  const current = INVENTORY[productId];
  INVENTORY[productId] = current + quantity;
  console.log('ADDED:' + INVENTORY[productId]);
}`
      },
      'state.js': {
        name: 'state.js',
        language: 'javascript',
        content: `export const INVENTORY = {
  'PROD_001': 100,
  'PROD_002': 50
};`
      },
      'test.js': {
        name: 'test.js',
        language: 'javascript',
        content: `import { decrementStock, addStock } from './inventory.js';
import { INVENTORY } from './state.js';

(async () => {
  // Start 3 concurrent decrements (should each decrement by 10, final = 70)
  await Promise.all([
    decrementStock('PROD_001', 10),
    decrementStock('PROD_001', 10),
    decrementStock('PROD_001', 10)
  ]);
  
  console.log('FINAL:' + INVENTORY['PROD_001']);
})();`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Race Condition Bug\n\nFix: Use a lock or queue to ensure only one operation modifies inventory at a time.\nExpected FINAL: 70 (100 - 30)\nActual: Will be 90 due to race condition\n\nSolution: Implement a lock mechanism or serial queue.'
      }
    },
    constraints: ['Serialize updates.', 'Use locking or queue pattern.'],
    testCheck: `
            const inventoryCode = files['inventory.js']?.content || '';
            const hasQueue = inventoryCode.includes('queue') || inventoryCode.includes('pending') || inventoryCode.includes('lock');
            const runtimePassed = stdout.includes('FINAL:70');
            return hasQueue && runtimePassed;
        `
  },
  {
    id: 'caching-invalidation-bug',
    category: 'backend',
    title: 'Stale Cache Not Invalidated',
    description: 'Debug a user profile system where cached data never expires or updates. Users see outdated information even after changes.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'userCache.js': {
        name: 'userCache.js',
        language: 'javascript',
        content: `import { CACHE_CONFIG } from './config.js';

const cache = {};

export function getUser(userId) {
  if (cache[userId]) {
    console.log('CACHE_HIT:' + userId);
    return cache[userId];
  }
  
  console.log('CACHE_MISS:' + userId);
  const user = { id: userId, name: 'John', email: 'john@example.com', updatedAt: Date.now() };
  
  // BUG: Caching without TTL or invalidation
  cache[userId] = user;
  return user;
}

export function updateUser(userId, updates) {
  // BUG: Modifying user but cache isn't invalidated
  const user = getUser(userId);
  Object.assign(user, updates, { updatedAt: Date.now() });
  
  console.log('UPDATED:' + userId);
  return user;
}

export function clearCache(userId) {
  // BUG: This is never called in updateUser
  delete cache[userId];
  console.log('CLEARED:' + userId);
}`
      },
      'config.js': {
        name: 'config.js',
        language: 'javascript',
        content: `export const CACHE_CONFIG = {
  TTL: 5000 // 5 seconds
};`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Cache Invalidation Bug\n\nFix: Invalidate cache when data is updated.\n- Call clearCache() in updateUser()\n- Or implement TTL-based expiration\n\nExpected: After update, next getUser() should show updated data'
      }
    },
    constraints: ['Invalidate on update.', 'Fresh data always returned after mutation.'],
    testCheck: `
            const cacheCode = files['userCache.js']?.content || '';
            const hasInvalidation = cacheCode.includes('clearCache');
            const runtimePassed = stdout.includes('CLEARED:');
            return hasInvalidation && runtimePassed;
        `
  },
  {
    id: 'string-escaping-xss',
    category: 'backend',
    title: 'XSS Vulnerability - Unescaped HTML',
    description: 'Debug a comment rendering system that injects malicious scripts. User input isn\'t sanitized before display.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'renderer.js': {
        name: 'renderer.js',
        language: 'javascript',
        content: `import { escapeHTML } from './sanitizer.js';

export function renderComment(comment) {
  // BUG: Directly inserting user input without escaping
  const html = '<div class="comment">' + comment.text + '</div>';
  console.log('RENDERED:' + html);
  return html;
}

export function renderCommentsSafe(comments) {
  const html = comments.map(c => {
    // TODO: Use escapeHTML here
    return '<p>' + c.text + '</p>';
  }).join('');
  
  console.log('SAFE_COUNT:' + comments.length);
  return html;
}`
      },
      'sanitizer.js': {
        name: 'sanitizer.js',
        language: 'javascript',
        content: `export function escapeHTML(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# XSS Vulnerability\n\nFix: Use escapeHTML() on all user inputs before rendering.\nMalicious input: `<script>alert("xss")</script>`\nShould render as: `&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;`'
      }
    },
    constraints: ['Escape all user input.', 'Use sanitizer module.'],
    testCheck: `
            const rendererCode = files['renderer.js']?.content || '';
            const hasSanitization = rendererCode.includes('escapeHTML');
            const runtimePassed = stdout.includes('&lt;') || stdout.includes('&amp;');
            return hasSanitization && runtimePassed;
        `
  },
  {
    id: 'event-listener-memory-leak',
    category: 'backend',
    title: 'Event Listeners Not Cleaned Up',
    description: 'Debug a real-time chat widget where event listeners accumulate on every page load, consuming memory and firing multiple times.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'chatWidget.js': {
        name: 'chatWidget.js',
        language: 'javascript',
        content: `import { createEmitter } from './eventEmitter.js';

const emitter = createEmitter();

export function initChat() {
  // BUG: Listener is added every time init is called, never removed
  emitter.on('message', (msg) => {
    console.log('MESSAGE_RECEIVED:' + msg);
  });
  
  console.log('CHAT_INIT:done');
}

export function sendMessage(text) {
  emitter.emit('message', text);
}

export function cleanup() {
  // BUG: This method exists but is never called
  emitter.removeAllListeners();
  console.log('LISTENERS_CLEARED');
}

export function getListenerCount() {
  return emitter.listenerCount('message');
}`
      },
      'eventEmitter.js': {
        name: 'eventEmitter.js',
        language: 'javascript',
        content: `export function createEmitter() {
  const listeners = {};
  
  return {
    on(event, handler) {
      if (!listeners[event]) listeners[event] = [];
      listeners[event].push(handler);
    },
    
    emit(event, data) {
      if (listeners[event]) {
        listeners[event].forEach(h => h(data));
      }
    },
    
    removeAllListeners() {
      Object.keys(listeners).forEach(e => listeners[e] = []);
    },
    
    listenerCount(event) {
      return listeners[event]?.length || 0;
    }
  };
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Event Listener Memory Leak\n\nFix: Call cleanup() when widget unmounts or add check to prevent duplicate listeners.\nBug: Calling initChat() twice results in 2 listeners for same event.'
      }
    },
    constraints: ['Prevent duplicate listeners.', 'Provide cleanup method.'],
    testCheck: `
            const chatCode = files['chatWidget.js']?.content || '';
            const hasCleanupCall = chatCode.includes('cleanup()') || chatCode.includes('removeAllListeners');
            const runtimePassed = stdout.includes('LISTENERS_CLEARED') || stdout.includes('listenerCount:1');
            return (hasCleanupCall || chatCode.includes('if.*listenerCount')) && runtimePassed;
        `
  },
  {
    id: 'null-pointer-chain',
    category: 'backend',
    title: 'Null Pointer Exception - Unsafe Property Access',
    description: 'Debug an API response handler that crashes when optional fields are missing. Deep property access without null checks.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'apiHandler.js': {
        name: 'apiHandler.js',
        language: 'javascript',
        content: `import { extractUserData } from './extractor.js';

export function handleUserResponse(response) {
  try {
    // BUG: Unsafe chaining - response.user might be null
    const name = response.user.profile.name;
    const email = response.user.contact.email;
    const phone = response.user.contact.phone || 'N/A';
    
    console.log('NAME:' + name);
    console.log('EMAIL:' + email);
    console.log('PHONE:' + phone);
    
    return { name, email, phone };
  } catch (err) {
    console.log('ERROR:' + err.message);
    return null;
  }
}

export function processUsers(responses) {
  return responses.map(r => {
    // BUG: No validation of r structure
    const user = extractUserData(r);
    return {
      fullName: user.firstName + ' ' + user.lastName,
      id: user.id
    };
  });
}`
      },
      'extractor.js': {
        name: 'extractor.js',
        language: 'javascript',
        content: `export function extractUserData(response) {
  // BUG: Assumes deep nesting exists without checking
  return {
    firstName: response.profile.personal.firstName,
    lastName: response.profile.personal.lastName,
    id: response.metadata.id,
    email: response.contact.email.primary
  };
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Null Pointer / Optional Chaining Bug\n\nFix: Use optional chaining (?.) or guard clauses.\nTest cases:\n1. Missing user object\n2. Missing nested profile\n3. Undefined email'
      }
    },
    constraints: ['Use optional chaining or guard clauses.', 'Handle missing fields gracefully.'],
    testCheck: `
            const code = (files['apiHandler.js']?.content || '') + (files['extractor.js']?.content || '');
            const hasOptionalChaining = code.includes('?.');
            const hasGuards = code.includes('?.') || code.includes('||') || code.includes('&&');
            return hasOptionalChaining || (hasGuards && !stdout.includes('ERROR:Cannot read'));
        `
  },
  {
    id: 'sql-injection-danger',
    category: 'backend',
    title: 'SQL Injection - Unsafe Query Building',
    description: 'Debug a user search feature vulnerable to SQL injection. String concatenation is used instead of parameterized queries.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'db.js': {
        name: 'db.js',
        language: 'javascript',
        content: `export function query(sql, params) {
  console.log('QUERY:' + sql);
  // Simulated DB response
  if (sql.includes('DROP') || sql.includes('DELETE')) {
    console.log('DANGER:Malicious query detected!');
    return [];
  }
  return { id: 1, name: 'Test' };
}`
      },
      'userRepo.js': {
        name: 'userRepo.js',
        language: 'javascript',
        content: `import { query } from './db.js';

export function findByEmail(email) {
  // BUG: SQL Injection vulnerability!
  const sql = 'SELECT * FROM users WHERE email = \'' + email + '\'';
  return query(sql, []);
}

export function searchUsers(searchTerm) {
  // BUG: Unsafe string concatenation
  const sql = 'SELECT * FROM users WHERE name LIKE \'%' + searchTerm + '%\'';
  return query(sql, []);
}

export function findByEmailSafe(email) {
  // TODO: Use parameterized query
  const sql = 'SELECT * FROM users WHERE email = \'' + email + '\'';
  return query(sql, []);
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# SQL Injection Bug\n\nFix: Use parameterized queries instead of string concatenation.\nMalicious payload: `\' OR \'1\'=\'1` \nSecure: `query(\'SELECT * FROM users WHERE email = ?\', [email])`'
      }
    },
    constraints: ['Use parameterized queries.', 'Never concatenate user input into SQL.'],
    testCheck: `
            const repoCode = files['userRepo.js']?.content || '';
            const hasParams = repoCode.includes('?') && repoCode.match(/query.*\\[.*\\]/);
            const noConcat = !repoCode.includes('\\'' + ' ');
            return hasParams && noConcat;
        `
  },
  {
    id: 'off-by-one-pagination',
    category: 'backend',
    title: 'Off-by-One Error in Pagination',
    description: 'Debug a data listing system where pagination boundaries are wrong. Page 2 shows duplicate or missing items.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'paginator.js': {
        name: 'paginator.js',
        language: 'javascript',
        content: `import { ITEMS } from './data.js';
import { PAGE_SIZE } from './config.js';

export function getPage(pageNumber) {
  // BUG: Off-by-one error in calculation
  const start = (pageNumber - 1) * PAGE_SIZE;
  const end = pageNumber * PAGE_SIZE;
  
  console.log('PAGE:' + pageNumber);
  console.log('START:' + start);
  console.log('END:' + end);
  
  const items = ITEMS.slice(start, end);
  console.log('COUNT:' + items.length);
  
  return items;
}

export function getTotalPages() {
  // BUG: Not rounding up correctly
  return ITEMS.length / PAGE_SIZE;
}

export function validatePageNumber(page, total) {
  // BUG: Off-by-one - should be <= total not < total
  if (page < 1 || page > total) {
    console.log('ERROR:Invalid page');
    return false;
  }
  return true;
}`
      },
      'data.js': {
        name: 'data.js',
        language: 'javascript',
        content: `export const ITEMS = Array.from({ length: 25 }, (_, i) => ({
  id: i + 1,
  name: 'Item ' + (i + 1)
}));`
      },
      'config.js': {
        name: 'config.js',
        language: 'javascript',
        content: `export const PAGE_SIZE = 10;`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Pagination Bug\n\n25 items, 10 per page = 3 pages\nPage 1: items 1-10\nPage 2: items 11-20\nPage 3: items 21-25\n\nFix off-by-one errors in validation and ceiling logic.'
      }
    },
    constraints: ['Correct math for pagination.', 'Use Math.ceil for total pages.'],
    testCheck: `
            const paginatorCode = files['paginator.js']?.content || '';
            const hasCeil = paginatorCode.includes('Math.ceil');
            const correctBounds = paginatorCode.includes('<= total') || paginatorCode.includes('Math.ceil');
            return hasCeil && correctBounds;
        `
  },
  {
    id: 'async-timing-bug',
    category: 'backend',
    title: 'Async/Promise Timing Issue',
    description: 'Debug a multi-step workflow where data is used before the async operation completes. Wrong promise chaining or missing await.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'workflow.js': {
        name: 'workflow.js',
        language: 'javascript',
        content: `import { fetchUserData, fetchUserPosts } from './api.js';

export async function getUserWithPosts(userId) {
  // BUG: Missing await - returns before data is fetched
  const user = fetchUserData(userId);
  const posts = fetchUserPosts(userId);
  
  console.log('USER_NAME:' + user.name); // undefined!
  console.log('POSTS_COUNT:' + posts.length); // undefined!
  
  return { user, posts };
}

export async function processUserWorkflow(userId) {
  // BUG: Not chaining promises correctly
  let user;
  fetchUserData(userId).then(u => {
    user = u;
  });
  
  console.log('STEP_1_COMPLETE');
  console.log('USER:' + user); // Still undefined
  
  // BUG: This executes too early
  const processedName = user.name.toUpperCase();
  console.log('PROCESSED:' + processedName);
}`
      },
      'api.js': {
        name: 'api.js',
        language: 'javascript',
        content: `export async function fetchUserData(userId) {
  await new Promise(r => setTimeout(r, 100));
  return { id: userId, name: 'Alice', email: 'alice@example.com' };
}

export async function fetchUserPosts(userId) {
  await new Promise(r => setTimeout(r, 100));
  return [{ id: 1, title: 'Post 1' }, { id: 2, title: 'Post 2' }];
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Async Timing Bug\n\nFix: Properly await async operations before using their results.\n- Use await in async functions\n- Don\'t forget Promise.all for parallel operations\n\nExpected: USER_NAME:Alice, POSTS_COUNT:2'
      }
    },
    constraints: ['Use proper await syntax.', 'Chain or parallelize promises correctly.'],
    testCheck: `
            const workflowCode = files['workflow.js']?.content || '';
            const hasAwait = workflowCode.includes('await');
            const runtimePassed = stdout.includes('USER_NAME:Alice') && stdout.includes('POSTS_COUNT:2');
            return hasAwait && runtimePassed;
        `
  },
  {
    id: 'case-sensitivity-bug',
    category: 'backend',
    title: 'Case Sensitivity Bug - String Comparison',
    description: 'Debug a role-based access system where roles aren\'t compared case-sensitively. "Admin" and "admin" are treated as different.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'auth.js': {
        name: 'auth.js',
        language: 'javascript',
        content: `import { ADMIN_ROLE, validateRole } from './roles.js';

export function checkAdminAccess(userRole) {
  // BUG: Case-sensitive comparison
  if (userRole === ADMIN_ROLE) {
    console.log('ACCESS:granted');
    return true;
  }
  
  console.log('ACCESS:denied');
  return false;
}

export function getUserPermissions(role) {
  // BUG: Role lookup is case-sensitive
  const perms = validateRole(role);
  console.log('PERMISSIONS:' + perms.length);
  return perms;
}`
      },
      'roles.js': {
        name: 'roles.js',
        language: 'javascript',
        content: `export const ADMIN_ROLE = 'admin';
export const USER_ROLE = 'user';

export function validateRole(role) {
  // BUG: Case sensitive check
  const validRoles = {
    admin: ['read', 'write', 'delete'],
    user: ['read']
  };
  
  return validRoles[role] || [];
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Case Sensitivity Bug\n\nFix: Normalize strings to lowercase before comparison.\nTest: User role = "ADMIN" should get full access\nProblem: Direct equality fails'
      }
    },
    constraints: ['Normalize to lowercase.', 'Ensure case-insensitive comparison.'],
    testCheck: `
            const authCode = files['auth.js']?.content || '';
            const rolesCode = files['roles.js']?.content || '';
            const hasLowerCase = authCode.includes('toLowerCase()') || rolesCode.includes('toLowerCase()');
            const runtimePassed = stdout.includes('ACCESS:granted');
            return hasLowerCase && runtimePassed;
        `
  },
  {
    id: 'type-coercion-surprise',
    category: 'backend',
    title: 'Type Coercion Bug - Loose Equality',
    description: 'Debug a validator that accepts invalid input due to JavaScript\'s loose equality. "0" == 0 == false causes logic errors.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'validator.js': {
        name: 'validator.js',
        language: 'javascript',
        content: `import { isValidScore, isValidStatus } from './checks.js';

export function validateProduct(product) {
  // BUG: Loose equality allows "0" to be treated as 0
  if (product.price == 0) {
    console.log('INVALID:Price cannot be 0');
    return false;
  }
  
  // BUG: isNaN might coerce string "123" to 123
  if (!isValidScore(product.rating)) {
    console.log('INVALID:Bad rating');
    return false;
  }
  
  console.log('VALID:Product accepted');
  return true;
}`
      },
      'checks.js': {
        name: 'checks.js',
        language: 'javascript',
        content: `export function isValidScore(score) {
  // BUG: This accepts strings that coerce to numbers
  return score >= 1 && score <= 5;
}

export function isValidStatus(status) {
  // BUG: Using == instead of ===
  if (status == 'active') {
    return true;
  }
  return false;
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Type Coercion Bug\n\nFix: Use strict equality (===) instead of loose (==).\nTest case: price = "0" should be invalid\n"0" == false is true in JS but should be rejected'
      }
    },
    constraints: ['Use strict equality (===).', 'Proper type checking.'],
    testCheck: `
            const validatorCode = files['validator.js']?.content || '';
            const checksCode = files['checks.js']?.content || '';
            const strict = validatorCode.includes('===') && checksCode.includes('===');
            const runtimePassed = stdout.includes('INVALID:Price cannot be 0');
            return strict && runtimePassed;
        `
  },
  {
    id: 'ecommerce-order-system',
    category: 'backend',
    title: 'E-Commerce Order Processing System',
    description: 'Fix a multi-tier order processing pipeline with bugs in inventory, payment, and shipping coordination. The system processes orders but fails to update inventory correctly, duplicate charges occur, and order status is inconsistent.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'orderService.js': {
        name: 'orderService.js',
        language: 'javascript',
        content: `import { inventoryManager } from './inventory.js';
import { paymentProcessor } from './payment.js';
import { notificationService } from './notifications.js';
import { generateOrderId } from './utils.js';

const orders = {}; // In-memory order store

export const orderService = {
  async createOrder(userId, items, shippingAddress) {
    const orderId = generateOrderId();
    
    // BUG 1: No validation of items or inventory check before payment
    const orderTotal = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    
    // BUG 2: Payment taken before inventory is reserved
    const paymentResult = await paymentProcessor.charge(userId, orderTotal);
    if (!paymentResult.success) {
      console.log('PAYMENT_FAILED:' + paymentResult.error);
      return null;
    }
    
    // BUG 3: Inventory update and payment not atomic - if inventory fails, payment isn't refunded
    const inventoryUpdate = await inventoryManager.reserveItems(items);
    if (!inventoryUpdate.success) {
      console.log('INVENTORY_FAILED:Items out of stock');
      // BUG: Should refund payment but doesn't
      return null;
    }
    
    const order = {
      id: orderId,
      userId,
      items,
      status: 'pending', // BUG 4: Never changes to confirmed
      total: orderTotal,
      createdAt: Date.now(),
      paymentId: paymentResult.id,
      shipping: { address: shippingAddress, status: 'pending' }
    };
    
    orders[orderId] = order;
    
    // BUG 5: Notification sent but doesn't verify delivery
    notificationService.sendOrderConfirmation(userId, orderId);
    
    console.log('ORDER_CREATED:' + orderId);
    return orderId;
  },
  
  getOrder(orderId) {
    // BUG 6: No authorization check - anyone can view any order
    return orders[orderId];
  },
  
  updateOrderStatus(orderId, status) {
    // BUG 7: Status can be set to invalid values, no state machine
    if (!orders[orderId]) return false;
    orders[orderId].status = status;
    return true;
  },
  
  async processRefund(orderId) {
    const order = orders[orderId];
    // BUG 8: Inventory is never restored when refunding
    const refund = await paymentProcessor.refund(order.paymentId, order.total);
    if (refund.success) {
      order.status = 'refunded';
      console.log('REFUND_PROCESSED:' + orderId);
    }
  }
};`
      },
      'inventory.js': {
        name: 'inventory.js',
        language: 'javascript',
        content: `const inventory = {
  'SKU-001': { name: 'Laptop', stock: 5, reserved: 0 },
  'SKU-002': { name: 'Mouse', stock: 50, reserved: 0 },
  'SKU-003': { name: 'Keyboard', stock: 20, reserved: 0 }
};

export const inventoryManager = {
  async reserveItems(items) {
    // BUG 1: No transaction - each item reserved separately, can fail mid-way
    for (const item of items) {
      const product = inventory[item.sku];
      
      // BUG 2: Using 'stock' directly without lock causes race conditions
      if (!product || (product.stock - product.reserved) < item.quantity) {
        return { success: false };
      }
      
      // BUG 3: Async operation without atomic update
      await new Promise(r => setTimeout(r, 10));
      product.reserved += item.quantity;
    }
    
    console.log('INVENTORY_RESERVED:success');
    return { success: true };
  },
  
  async confirmReservation(items) {
    // BUG 4: Never called, so stock is never actually decremented
    for (const item of items) {
      const product = inventory[item.sku];
      product.stock -= item.quantity;
      product.reserved -= item.quantity;
    }
  },
  
  async cancelReservation(items) {
    // BUG 5: Doesn't update stock, only unreserves
    for (const item of items) {
      const product = inventory[item.sku];
      product.reserved -= item.quantity;
    }
  },
  
  getInventory() {
    return inventory;
  }
};`
      },
      'payment.js': {
        name: 'payment.js',
        language: 'javascript',
        content: `const charges = {};
let chargeIdCounter = 1000;

export const paymentProcessor = {
  async charge(userId, amount) {
    // BUG 1: No idempotency - same request charged twice
    // Should use idempotency key
    
    // Simulate payment gateway call
    await new Promise(r => setTimeout(r, 50));
    
    // BUG 2: No validation of amount
    if (amount <= 0) {
      return { success: false, error: 'Invalid amount' };
    }
    
    const chargeId = 'CHARGE_' + (chargeIdCounter++);
    charges[chargeId] = {
      id: chargeId,
      userId,
      amount,
      status: 'succeeded', // BUG 3: Always succeeds, no actual validation
      timestamp: Date.now()
    };
    
    console.log('PAYMENT_CHARGED:' + chargeId);
    return { success: true, id: chargeId };
  },
  
  async refund(chargeId, amount) {
    // BUG 4: No check if charge exists or is already refunded
    // BUG 5: Partial refunds not tracked
    if (!charges[chargeId]) {
      return { success: false };
    }
    
    charges[chargeId].status = 'refunded';
    console.log('REFUND_ISSUED:' + chargeId);
    return { success: true };
  },
  
  getCharges() {
    return charges;
  }
};`
      },
      'notifications.js': {
        name: 'notifications.js',
        language: 'javascript',
        content: `const sentNotifications = [];

export const notificationService = {
  async sendOrderConfirmation(userId, orderId) {
    // BUG 1: Fire-and-forget, no error handling
    this.send(userId, 'ORDER_CONFIRMED', { orderId });
    // BUG 2: No tracking if notification was actually delivered
  },
  
  async sendShippingUpdate(userId, orderId, trackingId) {
    this.send(userId, 'SHIPPED', { orderId, trackingId });
  },
  
  send(userId, eventType, data) {
    // BUG 3: No validation of user existence
    const notification = {
      userId,
      type: eventType,
      data,
      sentAt: Date.now(),
      delivered: false // BUG 4: Never changes to true
    };
    
    sentNotifications.push(notification);
    console.log('NOTIFICATION_SENT:' + eventType);
    return notification;
  },
  
  getNotifications(userId) {
    // BUG 5: Includes undelivered and delivered without distinction
    return sentNotifications.filter(n => n.userId === userId);
  }
};`
      },
      'utils.js': {
        name: 'utils.js',
        language: 'javascript',
        content: `let orderIdCounter = 1;

export function generateOrderId() {
  // BUG: Simple counter, not unique across restarts
  // Should use timestamp or UUID
  return 'ORD-' + (orderIdCounter++);
}

export function validateEmail(email) {
  // BUG: Regex too permissive
  return /^.+@.+$/.test(email);
}

export function formatCurrency(amount) {
  // BUG: Not properly rounding
  return '$' + amount.toFixed(2);
}`
      },
      'integrationTest.js': {
        name: 'integrationTest.js',
        language: 'javascript',
        content: `import { orderService } from './orderService.js';
import { inventoryManager } from './inventory.js';
import { paymentProcessor } from './payment.js';

(async () => {
  console.log('=== Order System Test ===');
  
  const items = [
    { sku: 'SKU-001', quantity: 1, price: 1000 },
    { sku: 'SKU-002', quantity: 2, price: 50 }
  ];
  
  const orderId = await orderService.createOrder('user123', items, '123 Main St');
  
  if (orderId) {
    const order = orderService.getOrder(orderId);
    console.log('ORDER_TOTAL:' + order.total);
    console.log('ORDER_STATUS:' + order.status);
    console.log('INVENTORY:' + JSON.stringify(inventoryManager.getInventory()));
    console.log('CHARGES:' + Object.keys(paymentProcessor.getCharges()).length);
  }
  
  console.log('TEST_COMPLETE');
})();`
      }
    },
    constraints: ['Atomic transactions required.', 'Proper state management.', 'Error handling with rollback.'],
    testCheck: `
            const orderCode = files['orderService.js']?.content || '';
            const inventoryCode = files['inventory.js']?.content || '';
            const paymentCode = files['payment.js']?.content || '';
            
            const hasRefund = paymentCode.includes('refund') || orderCode.includes('refund');
            const hasConfirmation = inventoryCode.includes('confirmReservation') && orderCode.includes('confirmReservation');
            const runtimePassed = stdout.includes('ORDER_CREATED:') && stdout.includes('ORDER_TOTAL:1100');
            
            return hasRefund && hasConfirmation && runtimePassed;
        `
  },
  {
    id: 'realtime-messaging-app',
    category: 'backend',
    title: 'Real-Time Messaging Application',
    description: 'Debug a chat application with message ordering issues, missing messages, duplicate delivery, and user presence tracking bugs. Messages arrive out of order, some are lost, others duplicated.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'messageQueue.js': {
        name: 'messageQueue.js',
        language: 'javascript',
        content: `const messages = {}; // roomId -> message[]
let messageIdCounter = 1;

export const messageQueue = {
  async addMessage(roomId, userId, text, timestamp) {
    // BUG 1: messageId based on counter, can collide in distributed system
    const messageId = 'MSG_' + (messageIdCounter++);
    
    // BUG 2: No validation of roomId or userId
    if (!messages[roomId]) {
      messages[roomId] = [];
    }
    
    const message = {
      id: messageId,
      roomId,
      userId,
      text,
      // BUG 3: Using Date.now() instead of server timestamp
      // Client timestamps can be out of order
      timestamp: timestamp || Date.now(),
      delivered: false,
      read: false
    };
    
    // BUG 4: Push to array without ordering by timestamp
    messages[roomId].push(message);
    console.log('MESSAGE_ADDED:' + messageId);
    
    return message;
  },
  
  async getMessages(roomId, limit = 50) {
    // BUG 5: No pagination offset, always returns from start
    // BUG 6: Messages not sorted by timestamp
    const roomMessages = messages[roomId] || [];
    return roomMessages.slice(-limit);
  },
  
  async markAsDelivered(messageId) {
    // BUG 7: Inefficient linear search across all rooms
    for (const roomId in messages) {
      const msg = messages[roomId].find(m => m.id === messageId);
      if (msg) {
        msg.delivered = true;
        // BUG 8: Doesn't return, no confirmation
        return;
      }
    }
  },
  
  async markAsRead(messageId, userId) {
    // BUG 9: No validation that userId is in the room
    for (const roomId in messages) {
      const msg = messages[roomId].find(m => m.id === messageId);
      if (msg) {
        msg.read = true;
        return true;
      }
    }
  }
};`
      },
      'userPresence.js': {
        name: 'userPresence.js',
        language: 'javascript',
        content: `const presence = {}; // userId -> { status, rooms, lastSeen }

export const userPresence = {
  userJoinRoom(userId, roomId) {
    // BUG 1: No validation of userId or roomId format
    if (!presence[userId]) {
      presence[userId] = {
        status: 'online',
        rooms: [],
        lastSeen: Date.now()
      };
    }
    
    // BUG 2: No check for duplicate room, can add same room twice
    presence[userId].rooms.push(roomId);
    console.log('USER_JOINED:' + userId + ':' + roomId);
  },
  
  userLeaveRoom(userId, roomId) {
    // BUG 3: No error handling if user not in room
    const idx = presence[userId].rooms.indexOf(roomId);
    presence[userId].rooms.splice(idx, 1);
    
    // BUG 4: User should go offline only if no rooms left
    // Currently just leaves room but stays online
  },
  
  getRoomUsers(roomId) {
    // BUG 5: Very inefficient - O(n) for every room query
    const users = [];
    for (const userId in presence) {
      if (presence[userId].rooms.includes(roomId)) {
        users.push({ userId, status: presence[userId].status });
      }
    }
    return users;
  },
  
  userGoOffline(userId) {
    // BUG 6: Doesn't clear rooms, just marks offline
    // User still in rooms but offline
    if (presence[userId]) {
      presence[userId].status = 'offline';
      // BUG 7: No notification to other users
    }
  },
  
  getPresence(userId) {
    return presence[userId] || null;
  }
};`
      },
      'roomManager.js': {
        name: 'roomManager.js',
        language: 'javascript',
        content: `const rooms = {}; // roomId -> room data

export const roomManager = {
  createRoom(roomName, creatorId) {
    // BUG 1: No unique ID generation, uses name as ID
    // If 2 rooms created with same name, one overwrites
    const roomId = roomName.toLowerCase();
    
    if (rooms[roomId]) {
      console.log('ROOM_EXISTS:' + roomId);
      return null; // BUG 2: Creator probably should be added to existing room
    }
    
    rooms[roomId] = {
      id: roomId,
      name: roomName,
      createdAt: Date.now(),
      creatorId,
      members: [creatorId], // BUG 3: Creator not properly tracked
      maxMembers: 100
    };
    
    console.log('ROOM_CREATED:' + roomId);
    return roomId;
  },
  
  addUserToRoom(roomId, userId) {
    // BUG 4: No check if user already in room
    const room = rooms[roomId];
    if (!room) return false;
    
    if (room.members.length >= room.maxMembers) {
      console.log('ROOM_FULL:' + roomId);
      return false;
    }
    
    room.members.push(userId);
    console.log('USER_ADDED:' + userId);
    return true;
  },
  
  removeUserFromRoom(roomId, userId) {
    // BUG 5: No check if user was actually in room
    const room = rooms[roomId];
    const idx = room.members.indexOf(userId);
    room.members.splice(idx, 1);
    
    // BUG 6: Room should be deleted if empty, but isn't
  },
  
  getRoom(roomId) {
    return rooms[roomId];
  }
};`
      },
      'connectionHandler.js': {
        name: 'connectionHandler.js',
        language: 'javascript',
        content: `import { messageQueue } from './messageQueue.js';
import { userPresence } from './userPresence.js';
import { roomManager } from './roomManager.js';

export const connectionHandler = {
  async handleNewMessage(roomId, userId, text) {
    // BUG 1: Client timestamp used, not server timestamp
    const timestamp = Date.now();
    const message = await messageQueue.addMessage(roomId, userId, text, timestamp);
    
    // BUG 2: Should broadcast to all users in room, but doesn't
    // Message goes into queue but not sent to clients
    return message;
  },
  
  async handleUserJoin(roomId, userId) {
    // BUG 3: No atomic operation - multiple failures possible
    userPresence.userJoinRoom(userId, roomId);
    roomManager.addUserToRoom(roomId, userId);
    
    // BUG 4: History not sent to user
    // BUG 5: Presence not broadcast to others
    console.log('USER_CONNECTED:' + userId + ':' + roomId);
  },
  
  async handleUserLeave(roomId, userId) {
    userPresence.userLeaveRoom(userId, roomId);
    roomManager.removeUserFromRoom(roomId, userId);
    
    // BUG 6: No notification to remaining users
    console.log('USER_DISCONNECTED:' + userId + ':' + roomId);
  }
};`
      },
      'test.js': {
        name: 'test.js',
        language: 'javascript',
        content: `import { connectionHandler } from './connectionHandler.js';
import { messageQueue } from './messageQueue.js';
import { userPresence } from './userPresence.js';
import { roomManager } from './roomManager.js';

(async () => {
  console.log('=== Messaging App Test ===');
  
  // Create room
  const roomId = roomManager.createRoom('general', 'user1');
  
  // Users join
  await connectionHandler.handleUserJoin(roomId, 'user1');
  await connectionHandler.handleUserJoin(roomId, 'user2');
  
  // Send messages
  await connectionHandler.handleNewMessage(roomId, 'user1', 'Hello');
  await connectionHandler.handleNewMessage(roomId, 'user2', 'Hi there');
  await connectionHandler.handleNewMessage(roomId, 'user1', 'How are you?');
  
  // Check messages
  const msgs = await messageQueue.getMessages(roomId);
  console.log('MESSAGES_COUNT:' + msgs.length);
  
  // Check presence
  const roomUsers = userPresence.getRoomUsers(roomId);
  console.log('ROOM_USERS:' + roomUsers.length);
  
  console.log('TEST_COMPLETE');
})();`
      }
    },
    constraints: ['Proper message ordering.', 'Atomic user operations.', 'Presence tracking.'],
    testCheck: `
            const queueCode = files['messageQueue.js']?.content || '';
            const presenceCode = files['userPresence.js']?.content || '';
            const connCode = files['connectionHandler.js']?.content || '';
            
            const hasOrdering = queueCode.includes('sort') || queueCode.includes('timestamp');
            const hasPresence = presenceCode.includes('offlineonly') || presenceCode.includes('rooms.length === 0');
            const runtimePassed = stdout.includes('MESSAGES_COUNT:') && stdout.includes('ROOM_USERS:');
            
            return (hasOrdering || hasPresence) && runtimePassed;
        `
  },
  {
    id: 'user-auth-system',
    category: 'backend',
    title: 'User Authentication & Authorization',
    description: 'Debug a complete auth system with token expiration bugs, privilege escalation vulnerabilities, session hijacking issues, and password reset token reuse problems.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'authService.js': {
        name: 'authService.js',
        language: 'javascript',
        content: `import { tokenManager } from './tokenManager.js';
import { userStore } from './userStore.js';
import { passwordUtils } from './passwordUtils.js';

export const authService = {
  async register(email, password, name) {
    // BUG 1: No email validation, allows invalid formats
    if (!email || !password || password.length < 6) {
      console.log('INVALID_INPUT:Password too short');
      return null;
    }
    
    // BUG 2: No check for duplicate email
    const userId = Math.random().toString(36); // BUG 3: Not a real ID
    
    const passwordHash = passwordUtils.hashPassword(password);
    
    userStore.saveUser({
      id: userId,
      email,
      password: passwordHash,
      name,
      created: Date.now(),
      role: 'user', // BUG 4: Should never default to user
      suspended: false
    });
    
    console.log('USER_REGISTERED:' + userId);
    return userId;
  },
  
  async login(email, password) {
    const user = userStore.findByEmail(email);
    
    // BUG 5: User not found case doesn't fail, leaks user existence
    if (!user || !passwordUtils.verifyPassword(password, user.password)) {
      console.log('LOGIN_FAILED:Invalid credentials');
      return null;
    }
    
    // BUG 6: No check for suspended users
    // BUG 7: No login attempt rate limiting
    
    const token = tokenManager.createToken(user.id, user.role);
    console.log('LOGIN_SUCCESS:' + email);
    
    return { token, user: { id: user.id, email, name: user.name } };
  },
  
  async logout(token) {
    // BUG 8: Token never actually invalidated
    // Token can still be used after logout
    console.log('LOGOUT:' + token.substring(0, 10));
  }
};`
      },
      'tokenManager.js': {
        name: 'tokenManager.js',
        language: 'javascript',
        content: `const activeTokens = {};
const tokenBlacklist = []; // BUG: Never used for logout

export const tokenManager = {
  createToken(userId, role) {
    // BUG 1: Token format not secure (no signature)
    // Anyone can modify the payload
    const payload = {
      userId,
      role,
      exp: Date.now() + 3600000, // 1 hour
      iat: Date.now()
    };
    
    // BUG 2: Just base64 encoding, not encrypted
    const token = Buffer.from(JSON.stringify(payload)).toString('base64');
    activeTokens[token] = payload;
    
    console.log('TOKEN_CREATED:' + token.substring(0, 10));
    return token;
  },
  
  verifyToken(token) {
    // BUG 3: Blacklist never checked
    // BUG 4: No signature verification
    
    try {
      const payload = JSON.parse(Buffer.from(token, 'base64').toString());
      
      // BUG 5: Expiration check missing
      // if (payload.exp < Date.now()) return null;
      
      // BUG 6: Doesn't verify payload hasn't been modified
      return payload;
    } catch {
      return null;
    }
  },
  
  invalidateToken(token) {
    // BUG 7: Only adds to blacklist, but blacklist not checked in verify
    tokenBlacklist.push(token);
    console.log('TOKEN_INVALIDATED:' + token.substring(0, 10));
  }
};`
      },
      'authMiddleware.js': {
        name: 'authMiddleware.js',
        language: 'javascript',
        content: `import { tokenManager } from './tokenManager.js';

export function authMiddleware(req, res, next) {
  // BUG 1: Token taken from header without validation
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    console.log('NO_TOKEN:Unauthorized');
    return; // BUG 2: Should return error, not just log
  }
  
  const payload = tokenManager.verifyToken(token);
  
  if (!payload) {
    console.log('INVALID_TOKEN:Unauthorized');
    return; // BUG 3: Should prevent further processing
  }
  
  // BUG 4: No check if user still exists or is suspended
  req.userId = payload.userId;
  req.userRole = payload.role;
  
  next();
}

export function requireAdmin(req, res, next) {
  // BUG 5: Role can be spoofed by modifying token payload
  if (req.userRole !== 'admin') {
    console.log('FORBIDDEN:Admin required');
    return;
  }
  
  next();
}`
      },
      'passwordReset.js': {
        name: 'passwordReset.js',
        language: 'javascript',
        content: `import { userStore } from './userStore.js';
import { passwordUtils } from './passwordUtils.js';

const resetTokens = {}; // token -> { userId, expires, used }

export const passwordReset = {
  async requestReset(email) {
    const user = userStore.findByEmail(email);
    if (!user) {
      // BUG 1: Should not leak user existence
      console.log('USER_NOT_FOUND:' + email);
      return null;
    }
    
    // BUG 2: Token is predictable
    const resetToken = Math.random().toString(36).substring(7);
    
    resetTokens[resetToken] = {
      userId: user.id,
      // BUG 3: No expiration time set
      // BUG 4: Token never expires
      created: Date.now()
    };
    
    console.log('RESET_TOKEN_SENT:' + user.id);
    return resetToken;
  },
  
  async resetPassword(resetToken, newPassword) {
    // BUG 5: No check if token has already been used
    const tokenData = resetTokens[resetToken];
    
    if (!tokenData) {
      console.log('INVALID_TOKEN:Token not found');
      return false;
    }
    
    const user = userStore.findById(tokenData.userId);
    if (!user) return false;
    
    // BUG 6: No validation of new password strength
    const passwordHash = passwordUtils.hashPassword(newPassword);
    user.password = passwordHash;
    
    // BUG 7: Token not marked as used, can be reused
    // Should delete or mark as consumed
    console.log('PASSWORD_RESET:' + user.id);
    
    return true;
  }
};`
      },
      'userStore.js': {
        name: 'userStore.js',
        language: 'javascript',
        content: `const users = {};
const emailIndex = {};

export const userStore = {
  saveUser(user) {
    users[user.id] = user;
    emailIndex[user.email] = user.id;
  },
  
  findById(userId) {
    return users[userId];
  },
  
  findByEmail(email) {
    const userId = emailIndex[email.toLowerCase()]; // BUG: Case sensitivity
    return users[userId];
  },
  
  updateUser(userId, updates) {
    // BUG: No validation of what can be updated
    // User can change their own role
    Object.assign(users[userId], updates);
  },
  
  deleteUser(userId) {
    const user = users[userId];
    delete emailIndex[user.email];
    delete users[userId];
  }
};`
      },
      'passwordUtils.js': {
        name: 'passwordUtils.js',
        language: 'javascript',
        content: `export const passwordUtils = {
  hashPassword(password) {
    // BUG 1: Not actually hashing, just encoding
    // Should use bcrypt or similar
    return Buffer.from(password).toString('base64');
  },
  
  verifyPassword(password, hash) {
    // BUG 2: Vulnerable to timing attacks
    return Buffer.from(password).toString('base64') === hash;
  },
  
  isStrongPassword(password) {
    // BUG 3: Very weak validation
    return password.length >= 6; // Should require uppercase, numbers, special chars
  }
};`
      },
      'test.js': {
        name: 'test.js',
        language: 'javascript',
        content: `import { authService } from './authService.js';
import { tokenManager } from './tokenManager.js';
import { passwordReset } from './passwordReset.js';

(async () => {
  console.log('=== Auth System Test ===');
  
  // Register user
  const userId = await authService.register('user@example.com', 'password123', 'John');
  
  // Login
  const result = await authService.login('user@example.com', 'password123');
  if (result) {
    const token = result.token;
    console.log('LOGIN_SUCCESS:' + result.user.email);
    
    // Verify token
    const payload = tokenManager.verifyToken(token);
    console.log('TOKEN_VALID:' + !!payload);
    
    // Logout (should invalidate token)
    await authService.logout(token);
    
    // BUG: Token still works after logout
    const stillValid = tokenManager.verifyToken(token);
    console.log('AFTER_LOGOUT:' + !!stillValid);
  }
  
  console.log('TEST_COMPLETE');
})();`
      }
    },
    constraints: ['Proper token validation.', 'Password security.', 'Token expiration.'],
    testCheck: `
            const tokenCode = files['tokenManager.js']?.content || '';
            const authCode = files['authService.js']?.content || '';
            const resetCode = files['passwordReset.js']?.content || '';
            
            const hasExpiration = tokenCode.includes('exp') && (authCode.includes('exp') || resetCode.includes('expires'));
            const hasBlacklist = tokenCode.includes('blacklist') && authCode.includes('invalidateToken');
            const runtimePassed = stdout.includes('LOGIN_SUCCESS:') && stdout.includes('TOKEN_VALID:');
            
            return hasExpiration && hasBlacklist && runtimePassed;
        `
  },
  {
    id: 'conduit-article-feed',
    category: 'backend',
    title: 'Conduit: Global Article Feed',
    description: 'Implement a high-performance data transformation layer for the Conduit (RealWorld) article feed. Map raw database records to the standardized API response structure.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'mockDatabase.js': {
        name: 'mockDatabase.js',
        language: 'javascript',
        content: `export const RAW_ARTICLES = [
  {
    id: 1,
    slug: "how-to-train-your-dragon",
    title: "How to train your dragon",
    description: "Ever wonder how?",
    body: "It takes a lot of patience and berries.",
    tags: ["dragons", "training"],
    created_at: "2016-02-18T03:22:56.637Z",
    updated_at: "2016-02-18T03:48:35.824Z",
    author_id: 101,
    fav_count: 42
  },
  {
    id: 2,
    slug: "how-to-train-your-dragon-2",
    title: "How to train your dragon 2",
    description: "So toothless",
    body: "It is a dragon.",
    tags: ["dragons", "training"],
    created_at: "2016-02-18T03:30:56.637Z",
    updated_at: "2016-02-18T03:50:35.824Z",
    author_id: 101,
    fav_count: 10
  }
];

export const RAW_USERS = {
  101: {
    username: "jake",
    bio: "I work at statefarm",
    image: "https://i.stack.imgur.com/xHW68.jpg",
    following: false
  }
};`
      },
      'apiContract.js': {
        name: 'apiContract.js',
        language: 'javascript',
        content: `/**
 * Expected Output Format per RealWorld Spec:
 * {
 *   "articles": [{
 *     "slug": string,
 *     "title": string,
 *     "description": string,
 *     "body": string,
 *     "tagList": string[],
 *     "createdAt": string (ISO),
 *     "updatedAt": string (ISO),
 *     "favorited": boolean,
 *     "favoritesCount": number,
 *     "author": {
 *       "username": string,
 *       "bio": string,
 *       "image": string,
 *       "following": boolean
 *     }
 *   }],
 *   "articlesCount": number
 * }
 */`
      },
      'feedTransformer.js': {
        name: 'feedTransformer.js',
        language: 'javascript',
        content: `import { RAW_ARTICLES, RAW_USERS } from './mockDatabase.js';

export function transformFeed(articles, users, currentUserId) {
  // TODO: Transform the raw database articles into the API contract format
  // 1. Map 'tags' to 'tagList'
  // 2. camelCase the timestamps
  // 3. Nest the author object using author_id
  // 4. Calculate 'favorited' status (assume false for now)
  
  return {
    articles: articles.map(a => {
      return a; // BUG: Returning untransformed data
    }),
    articlesCount: articles.length
  };
}`
      },
      'index.js': {
        name: 'index.js',
        language: 'javascript',
        content: `import { transformFeed } from './feedTransformer.js';
import { RAW_ARTICLES, RAW_USERS } from './mockDatabase.js';

const result = transformFeed(RAW_ARTICLES, RAW_USERS, null);
console.log('ARTICLES_COUNT:' + result.articlesCount);
if (result.articles[0] && result.articles[0].author) {
  console.log('AUTHOR_NAME:' + result.articles[0].author.username);
  console.log('TAG_LIST:' + (result.articles[0].tagList ? 'VALID' : 'MISSING'));
  console.log('CAMEL_CASE:' + (result.articles[0].createdAt ? 'VALID' : 'MISSING'));
}`
      },
      'README.md': {
        name: 'README.md',
        language: 'markdown',
        content: '# Conduit: Global Article Feed\n\nYour task is to implement the `transformFeed` function in `feedTransformer.js` to match the **RealWorld (Conduit)** API specification.\n\n### Requirements:\n- Rename `tags` to `tagList`.\n- Convert `created_at` and `updated_at` to `createdAt` and `updatedAt`.\n- Rename `fav_count` to `favoritesCount`.\n- Populate the `author` object using the `author_id` to lookup data in the `users` map.\n- Add a `favorited` boolean (default to `false`).'
      }
    },
    constraints: ['Strict adherence to OpenAPI schema.', 'Efficient map-lookup for authors.', 'Correct timestamp mapping.'],
    testCheck: `
            const transformerCode = files['feedTransformer.js']?.content || '';
            const hasAuthorMapping = transformerCode.includes('RAW_USERS[') || transformerCode.includes('users[');
            const hasTagMapping = transformerCode.includes('tagList');
            const hasCamelCase = transformerCode.includes('createdAt');
            const runtimePassed = stdout.includes('ARTICLES_COUNT:2') && stdout.includes('AUTHOR_NAME:jake') && stdout.includes('TAG_LIST:VALID') && stdout.includes('CAMEL_CASE:VALID');
            
            return hasAuthorMapping && hasTagMapping && hasCamelCase && runtimePassed;
        `
  },

  // ─── FRONTEND SCENARIOS ───────────────────────────────────────────────────
  {
    id: 'css-flexbox-bug',
    category: 'frontend',
    type: 'frontend',
    title: 'Broken Flex Layout',
    description: 'Three nav items should sit side-by-side in a horizontal row, but they are stacking vertically. Fix the CSS so the layout displays correctly.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .navbar {
      display: flex;
      /* BUG: items are stacking because of wrong flex-direction */
      flex-direction: column;
      gap: 12px;
      background: #1e1e1e;
      padding: 16px 32px;
      border-radius: 12px;
      border: 1px solid #333;
    }
    .nav-item {
      padding: 10px 24px;
      background: #7c3aed;
      border-radius: 8px;
      font-weight: bold;
      color: #fff;
      text-align: center;
    }
  </style>
</head>
<body>
  <nav class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </nav>
</body>
</html>`
      }
    },
    constraints: ['Items must use flexbox.', "Change flex-direction to 'row'."],
    desiredOutput: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .navbar { display: flex; flex-direction: row; gap: 12px; background: #1e1e1e; padding: 16px 32px; border-radius: 12px; border: 1px solid #333; }
    .nav-item { padding: 10px 24px; background: #7c3aed; border-radius: 8px; font-weight: bold; color: #fff; text-align: center; }
  </style>
</head>
<body>
  <nav class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">About</div>
    <div class="nav-item">Contact</div>
  </nav>
</body>
</html>`,
    previewCheck: `
      try {
        const navbar = document.querySelector('.navbar');
        const style = window.getComputedStyle(navbar);
        const dir = style.flexDirection;
        return dir === 'row' || dir === 'row-reverse';
      } catch(e) { return false; }
    `
  },
  {
    id: 'js-counter-bug',
    category: 'frontend',
    type: 'frontend',
    title: 'Click Counter Bug',
    description: 'A counter button should increment the displayed number each time it is clicked. The counter is stuck at 0. Find and fix the two bugs.',
    difficulty: 'Easy',
    badgeColor: 'bg-green-500/10 text-green-500 border border-green-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    #count { font-size: 72px; font-weight: 900; color: #a78bfa; }
    button { padding: 14px 40px; background: #7c3aed; color: #fff; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; }
    button:hover { background: #6d28d9; }
  </style>
</head>
<body>
  <div id="count">0</div>
  <!-- BUG 1: button type submit causes page reload -->
  <button type="submit" id="btn">Click Me</button>
  <script>
    let count = 0;
    const display = document.getElementById('count');
    // BUG 2: wrong element ID - 'button' does not exist, should be 'btn'
    document.getElementById('button').addEventListener('click', () => {
      count++;
      display.textContent = count;
    });
  <\/script>
</body>
</html>`
      }
    },
    constraints: ["Fix button type to 'button'.", "Fix getElementById to use 'btn'."],
    desiredOutput: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    #count { font-size: 72px; font-weight: 900; color: #a78bfa; }
    button { padding: 14px 40px; background: #7c3aed; color: #fff; border: none; border-radius: 12px; font-size: 18px; font-weight: bold; cursor: pointer; }
  </style>
</head>
<body>
  <div id="count">1</div>
  <button type="button" id="btn">Click Me</button>
</body>
</html>`,
    previewCheck: `
      try {
        const btn = document.getElementById('btn');
        if (!btn) return false;
        const correctType = btn.type === 'button';
        btn.click();
        const count = parseInt(document.getElementById('count').textContent, 10);
        return correctType && count > 0;
      } catch(e) { return false; }
    `
  },
  {
    id: 'dom-color-bug',
    category: 'frontend',
    type: 'frontend',
    title: 'Dynamic Color Picker',
    description: 'Clicking a color swatch should change the background of the box below it. Nothing happens. Fix the JavaScript typo.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 32px; }
    .swatches { display: flex; gap: 16px; }
    .swatch { width: 60px; height: 60px; border-radius: 50%; cursor: pointer; border: 3px solid transparent; transition: transform 0.2s; }
    .swatch:hover { transform: scale(1.1); border-color: #fff; }
    #color-box { width: 200px; height: 200px; border-radius: 24px; background: #1e1e1e; border: 2px solid #333; transition: background 0.3s; }
    p { color: #666; font-size: 14px; }
  </style>
</head>
<body>
  <div class="swatches">
    <div class="swatch" data-color="#7c3aed" style="background:#7c3aed"></div>
    <div class="swatch" data-color="#ec4899" style="background:#ec4899"></div>
    <div class="swatch" data-color="#10b981" style="background:#10b981"></div>
    <div class="swatch" data-color="#f59e0b" style="background:#f59e0b"></div>
  </div>
  <div id="color-box"></div>
  <p>Click a color swatch above</p>
  <script>
    const swatches = document.querySelectorAll('.swatch');
    const box = document.getElementById('color-box');
    swatches.forEach(swatch => {
      swatch.addEventListener('click', () => {
        // BUG: dataset.colour does not exist (typo - should be dataset.color)
        box.style.backgroundColor = swatch.dataset.colour;
      });
    });
  <\/script>
</body>
</html>`
      }
    },
    constraints: ["Fix 'dataset.colour' typo to 'dataset.color'.", "Set box.style.backgroundColor."],
    desiredOutput: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 32px; }
    .swatches { display: flex; gap: 16px; }
    .swatch { width: 60px; height: 60px; border-radius: 50%; cursor: pointer; border: 3px solid transparent; }
    #color-box { width: 200px; height: 200px; border-radius: 24px; background: #7c3aed; border: 2px solid #7c3aed; }
  </style>
</head>
<body>
  <div class="swatches">
    <div class="swatch" style="background:#7c3aed"></div>
    <div class="swatch" style="background:#ec4899"></div>
  </div>
  <div id="color-box" style="background:#7c3aed"></div>
</body>
</html>`,
    previewCheck: `
      try {
        const swatch = document.querySelector('.swatch');
        const box = document.getElementById('color-box');
        if (!swatch || !box) return false;
        swatch.click();
        const bg = box.style.backgroundColor || box.style.background;
        return bg !== '' && bg !== undefined && bg !== 'rgba(0, 0, 0, 0)';
      } catch(e) { return false; }
    `
  },
  {
    id: 'css-card-hover-bug',
    category: 'frontend',
    type: 'frontend',
    title: 'Card Hover Effect Bug',
    description: 'Three product cards should lift and glow on hover. The hover animation is completely broken. Fix the two CSS bugs.',
    difficulty: 'Medium',
    badgeColor: 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    .card {
      width: 180px; padding: 28px 20px; background: #1a1a2e;
      border: 1px solid #333; border-radius: 16px; text-align: center; cursor: pointer;
      /* BUG 1: transition is missing */
    }
    /* BUG 2: wrong selector - '.cards:hover' should be '.card:hover' */
    .cards:hover {
      transform: translateY(-12px);
      box-shadow: 0 20px 60px rgba(124, 58, 237, 0.5);
      border-color: #7c3aed;
    }
    .card-icon { font-size: 40px; margin-bottom: 12px; }
    .card-title { font-size: 14px; font-weight: bold; color: #a78bfa; }
    .card-price { font-size: 22px; font-weight: 900; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card"><div class="card-icon">&#127911;</div><div class="card-title">Headphones</div><div class="card-price">$129</div></div>
  <div class="card"><div class="card-icon">&#9000;</div><div class="card-title">Keyboard</div><div class="card-price">$89</div></div>
  <div class="card"><div class="card-icon">&#128432;</div><div class="card-title">Mouse</div><div class="card-price">$59</div></div>
</body>
</html>`
      }
    },
    constraints: ["Add 'transition: transform 0.3s, box-shadow 0.3s' to .card.", "Fix hover selector from '.cards:hover' to '.card:hover'."],
    desiredOutput: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: sans-serif; background: #0a0a0a; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; gap: 24px; }
    .card { width: 180px; padding: 28px 20px; background: #1a1a2e; border: 1px solid #7c3aed; border-radius: 16px; text-align: center; cursor: pointer; transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s; transform: translateY(-12px); box-shadow: 0 20px 60px rgba(124,58,237,0.5); }
    .card-icon { font-size: 40px; margin-bottom: 12px; }
    .card-title { font-size: 14px; font-weight: bold; color: #a78bfa; }
    .card-price { font-size: 22px; font-weight: 900; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card"><div class="card-icon">&#127911;</div><div class="card-title">Headphones</div><div class="card-price">$129</div></div>
  <div class="card"><div class="card-icon">&#9000;</div><div class="card-title">Keyboard</div><div class="card-price">$89</div></div>
  <div class="card"><div class="card-icon">&#128432;</div><div class="card-title">Mouse</div><div class="card-price">$59</div></div>
</body>
</html>`,
    previewCheck: `
      try {
        const card = document.querySelector('.card');
        if (!card) return false;
        const style = window.getComputedStyle(card);
        const hasTransition = style.transitionDuration && style.transitionDuration !== '0s';
        let hasHoverRule = false;
        for (const sheet of document.styleSheets) {
          try {
            for (const rule of sheet.cssRules) {
              if (rule.selectorText && rule.selectorText.includes('.card:hover')) hasHoverRule = true;
            }
          } catch(e) {}
        }
        return hasTransition && hasHoverRule;
      } catch(e) { return false; }
    `
  },
  {
    id: 'js-todo-list-bug',
    category: 'frontend',
    type: 'frontend',
    title: 'Todo List Builder Bug',
    description: 'A todo app should add new tasks when clicking "Add". Items are not appearing and the input is never cleared. Fix the three JavaScript bugs.',
    difficulty: 'Hard',
    badgeColor: 'bg-red-500/10 text-red-500 border border-red-500/20',
    files: {
      'index.html': {
        name: 'index.html',
        language: 'html',
        content: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #0d0d0d; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .app { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; padding: 32px; width: 380px; }
    h2 { font-size: 22px; font-weight: 900; margin-bottom: 24px; color: #a78bfa; }
    .input-row { display: flex; gap: 10px; margin-bottom: 24px; }
    input { flex: 1; padding: 12px 16px; background: #111; border: 1px solid #333; border-radius: 10px; color: #fff; font-size: 14px; outline: none; }
    input:focus { border-color: #7c3aed; }
    button { padding: 12px 20px; background: #7c3aed; border: none; border-radius: 10px; color: #fff; font-weight: bold; cursor: pointer; font-size: 14px; }
    button:hover { background: #6d28d9; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    li { padding: 14px 18px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
    li::before { content: '\\2713'; color: #7c3aed; font-weight: bold; }
  </style>
</head>
<body>
  <div class="app">
    <h2>My Todos</h2>
    <div class="input-row">
      <input type="text" id="todo-input" placeholder="Add a task..." />
      <button id="add-btn">Add</button>
    </div>
    <ul id="todo-list"></ul>
  </div>
  <script>
    const input = document.getElementById('todo-input');
    const list = document.getElementById('todo-list');
    const btn = document.getElementById('add-btn');
    btn.addEventListener('click', addTodo);
    function addTodo() {
      const text = input.value;
      // BUG 1: No empty check - should guard with: if (!text.trim()) return;
      const li = document.createElement('li');
      // BUG 2: Using innerHTML with raw text - should use textContent
      li.innerHTML = text;
      list.appendChild(li);
      // BUG 3: Input is never cleared - should add: input.value = '';
    }
  <\/script>
</body>
</html>`
      }
    },
    constraints: ['Guard against empty input with trim() check.', 'Use textContent not innerHTML.', 'Clear input.value after adding.'],
    desiredOutput: `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; background: #0d0d0d; color: #fff; display: flex; align-items: center; justify-content: center; height: 100vh; }
    .app { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 20px; padding: 32px; width: 380px; }
    h2 { font-size: 22px; font-weight: 900; margin-bottom: 24px; color: #a78bfa; }
    .input-row { display: flex; gap: 10px; margin-bottom: 24px; }
    input { flex: 1; padding: 12px 16px; background: #111; border: 1px solid #333; border-radius: 10px; color: #fff; font-size: 14px; }
    button { padding: 12px 20px; background: #7c3aed; border: none; border-radius: 10px; color: #fff; font-weight: bold; cursor: pointer; }
    ul { list-style: none; display: flex; flex-direction: column; gap: 10px; }
    li { padding: 14px 18px; background: #111; border: 1px solid #2a2a2a; border-radius: 10px; font-size: 14px; display: flex; align-items: center; gap: 12px; }
    li::before { content: '\\2713'; color: #7c3aed; font-weight: bold; }
  </style>
</head>
<body>
  <div class="app">
    <h2>My Todos</h2>
    <div class="input-row">
      <input type="text" id="todo-input" placeholder="Add a task..." />
      <button id="add-btn">Add</button>
    </div>
    <ul id="todo-list">
      <li>Buy groceries</li>
      <li>Write clean code</li>
    </ul>
  </div>
</body>
</html>`,
    previewCheck: `
      try {
        const input = document.getElementById('todo-input');
        const btn = document.getElementById('add-btn');
        const list = document.getElementById('todo-list');
        if (!input || !btn || !list) return false;
        input.value = 'Test Task';
        btn.click();
        const items = list.querySelectorAll('li');
        const added = items.length > 0;
        const cleared = input.value === '';
        const safeContent = items.length > 0 && items[items.length-1].textContent.trim() === 'Test Task';
        return added && cleared && safeContent;
      } catch(e) { return false; }
    `
  },
  {
    "id": "dynamic-form-builder",
    "title": "Dynamic Form Builder (Missing unique ID bug)",
    "description": "The dynamic form builder is generating forms with missing unique IDs, causing issues with form submission and validation.",
    "difficulty": "Medium",
    "badgeColor": "bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",
    "files": {
      "index.js": {
        "name": "index.js",
        "language": "javascript",
        "content": "const FormBuilder = function() {\n  let forms = [];\n  return {\n    addForm: function(form) {\n      forms.push(form);\n      return this;\n    },\n    getForms: function() {\n      return forms;\n    }\n  };\n}\n\nconst formBuilder = FormBuilder();\nformBuilder.addForm({ id: 1, name: 'Form 1' });\nformBuilder.addForm({ id: 2, name: 'Form 2' });\nconsole.log(formBuilder.getForms());"
      },
      "README.md": {
        "name": "README.md",
        "language": "markdown",
        "content": "# Dynamic Form Builder (Missing unique ID bug)\n\n## Task\n\nImplement a dynamic form builder that adds a unique ID to each form.\n\n## Constraints\n\n* Each form must have a unique ID.\n* The form builder must be reusable."
      }
    },
    "constraints": ["Each form must have a unique ID.", "The form builder must be reusable."],
    "testCheck": "files.getForms().every(form => form.id !== undefined)"
  },
  {
    "id": "infinite-scroll",
    "title": "Infinite Scroll (Double-fetch race condition)",
    "description": "The infinite scroll feature is causing a double-fetch race condition, resulting in duplicate data being fetched and displayed.",
    "difficulty": "Hard",
    "badgeColor": "bg-red-500/10 text-red-500 border border-red-500/20",
    "files": {
      "index.js": {
        "name": "index.js",
        "language": "javascript",
        "content": "const InfiniteScroll = function() {\n  let data = [];\n  let isLoading = false;\n  return {\n    fetchData: function() {\n      if (isLoading) return;\n      isLoading = true;\n      // Simulate data fetching\n      setTimeout(() => {\n        data.push({ id: 1, name: 'Item 1' });\n        data.push({ id: 2, name: 'Item 2' });\n        isLoading = false;\n        console.log(data);\n      }, 1000);\n    }\n  };\n}\n\nconst infiniteScroll = InfiniteScroll();\ninfiniteScroll.fetchData();\ninfiniteScroll.fetchData();"
      },
      "README.md": {
        "name": "README.md",
        "language": "markdown",
        "content": "# Infinite Scroll (Double-fetch race condition)\n\n## Task\n\nImplement an infinite scroll feature that fetches data without causing a double-fetch race condition.\n\n## Constraints\n\n* The feature must fetch data without causing a double-fetch race condition.\n* The feature must be reusable."
      }
    },
    "constraints": ["The feature must fetch data without causing a double-fetch race condition.", "The feature must be reusable."],
    "testCheck": "files.infiniteScroll.isLoading === false"
  },
  {
    "id": "deep-clone-utility",
    "title": "Deep Clone Utility (Recursive reference bug)",
    "description": "The deep clone utility is causing a recursive reference bug, resulting in an infinite loop when trying to clone complex objects.",
    "difficulty": "Easy",
    "badgeColor": "bg-green-500/10 text-green-500 border border-green-500/20",
    "files": {
      "index.js": {
        "name": "index.js",
        "language": "javascript",
        "content": "const clone = function(obj) {\n  return JSON.parse(JSON.stringify(obj));\n}\n\nconst original = {\n  a: 1,\n  b: 2,\n  c: {\n    d: 3,\n    e: 4\n  }\n};\n\nconst cloned = clone(original);\nconsole.log(cloned);"
      },
      "README.md": {
        "name": "README.md",
        "language": "markdown",
        "content": "# Deep Clone Utility (Recursive reference bug)\n\n## Task\n\nImplement a deep clone utility that clones complex objects without causing a recursive reference bug.\n\n## Constraints\n\n* The utility must clone complex objects without causing a recursive reference bug.\n* The utility must be reusable."
      }
    },
    "constraints": ["The utility must clone complex objects without causing a recursive reference bug.", "The utility must be reusable."],
    "testCheck": "JSON.stringify(files.original) === JSON.stringify(files.cloned)"
  },
  {
    "id": "css-filter-service",
    "title": "CSS Filter Service (Color parsing bug)",
    "description": "The CSS filter service is causing a color parsing bug, resulting in incorrect color values being returned.",
    "difficulty": "Medium",
    "badgeColor": "bg-orange-500/10 text-orange-500 border border-orange-500/20",
    "files": {
      "index.js": {
        "name": "index.js",
        "language": "javascript",
        "content": "const parseColor = function(color) {\n  return color.match(/^#([0-9a-f]{3}){1,2}$/i);\n}\n\nconst color = '#123456';\nconsole.log(parseColor(color));"
      },
      "README.md": {
        "name": "README.md",
        "language": "markdown",
        "content": "# CSS Filter Service (Color parsing bug)\n\n## Task\n\nImplement a CSS filter service that parses color values without causing a color parsing bug.\n\n## Constraints\n\n* The service must parse color values without causing a color parsing bug.\n* The service must be reusable."
      }
    },
    "constraints": ["The service must parse color values without causing a color parsing bug.", "The service must be reusable."],
    "testCheck": "files.parseColor('#123456') === null"
  },
  {
    "id": "log-parser",
    "title": "Log Parser (Complex Regex challenge)",
    "description": "The log parser is struggling with a complex regex challenge, resulting in incorrect log parsing.",
    "difficulty": "Hard",
    "badgeColor": "bg-blue-500/10 text-blue-500 border border-blue-500/20",
    "files": {
      "index.js": {
        "name": "index.js",
        "language": "javascript",
        "content": "const parseLog = function(log) {\n  const regex = /^([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}) \\[(INFO|ERROR|WARNING)\\] (.*)$/;\n  return log.match(regex);\n}\n\nconst log = '2022-01-01 12:00:00 [INFO] This is a log message.';\nconsole.log(parseLog(log));"
      },
      "README.md": {
        "name": "README.md",
        "language": "markdown",
        "content": "# Log Parser (Complex Regex challenge)\n\n## Task\n\nImplement a log parser that parses log messages with a complex regex.\n\n## Constraints\n\n* The parser must parse log messages with the complex regex.\n* The parser must be reusable."
      }
    },
    "constraints": ["The parser must parse log messages with the complex regex.", "The parser must be reusable."],
    "testCheck": "files.parseLog('2022-01-01 12:00:00 [INFO] This is a log message.') !== null"
  }
];
