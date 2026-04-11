import { authService } from './authService.js';
import { tokenManager } from './tokenManager.js';
import { passwordReset } from './passwordReset.js';

(async () => {
    console.log('=== Auth System Test ===');

    // Register user
    const userId = await authService.register('user@example.com', 'Password!123', 'John');

    // Login
    const result = await authService.login('user@example.com', 'Password!123');
    if (result) {
        const token = result.token;
        console.log('LOGIN_SUCCESS:' + result.user.email);

        // Verify token
        const payload = tokenManager.verifyToken(token);
        console.log('TOKEN_VALID:' + !!payload);

        // Logout (should invalidate token)
        await authService.logout(token);

        // Token should NOT work after logout
        const stillValid = tokenManager.verifyToken(token);
        console.log('AFTER_LOGOUT:' + !!stillValid);
    }

    console.log('TEST_COMPLETE');
})();
