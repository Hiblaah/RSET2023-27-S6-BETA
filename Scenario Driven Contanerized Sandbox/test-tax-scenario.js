const url = 'http://localhost:3001/execute';

const payload = {
  files: {
    'index.js': {
      content: "import { ShoppingCart } from './cart.js';\nconst cart = new ShoppingCart(0.1); // 10% tax\ncart.addItem({ price: 100, quantity: 1 });\nconst total = cart.calculateTotal();\nconsole.log('RESULT_TOTAL:' + total);"
    },
    'cart.js': {
      content: "export class ShoppingCart {\n  constructor(taxRate = 0.08) {\n    this.items = [];\n    this.taxRate = taxRate;\n  }\n\n  addItem(item) {\n    this.items.push(item);\n  }\n\n  calculateTotal() {\n    const subtotal = this.items.reduce((sum, item) => sum + (item.price * item.quantity), 0);\n    return subtotal * (1 + this.taxRate);\n  }\n}"
    }
  },
  entryPoint: 'index.js',
  testCheck: "const cartCode = files['cart.js']?.content || '';\nconst hasFormula = cartCode.includes('taxRate') && (cartCode.includes('*') || cartCode.includes('+'));\nconst runtimePassed = stdout.includes('RESULT_TOTAL:110');\nreturn hasFormula && runtimePassed;"
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
}).then(r => r.json()).then(console.log).catch(console.error);
