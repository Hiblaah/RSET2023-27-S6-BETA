const { executeCode } = require('./executor');

(async () => {
  console.log('Testing Docker execution...\n');
  
  const result = await executeCode(
    { 'index.js': { content: 'console.log("Docker is working!");' } },
    'index.js'
  );
  
  console.log('=== RESULT ===');
  console.log('stdout:', result.stdout.trim());
  console.log('stderr:', result.stderr.trim());
  console.log('exitCode:', result.exitCode);
  console.log('timedOut:', result.timedOut);
  
  if (result.exitCode === 0 && result.stdout.includes('Docker is working')) {
    console.log('\n✅ Docker test PASSED!');
  } else {
    console.log('\n❌ Docker test FAILED!');
  }
})();
