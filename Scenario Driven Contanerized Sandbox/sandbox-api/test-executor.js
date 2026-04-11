const { executeCode } = require('./executor');

(async () => {
  console.log('Testing executor...');
  const result = await executeCode(
    { 'index.js': { content: 'console.log("Hello World");' } },
    'index.js'
  );
  console.log('=== RESULT ===');
  console.log(JSON.stringify(result, null, 2));
})();
