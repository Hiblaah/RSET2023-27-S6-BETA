# Fix Email Rate Limit Error

## The Problem
You're seeing: **"Email rate limit exceeded"**

This happens because Supabase sends confirmation emails when users sign up, and the free tier has strict rate limits.

## Solution: Disable Email Confirmation (For Development)

### Option 1: Disable in Supabase Dashboard (Recommended for Development)

1. Go to your Supabase project: https://app.supabase.com/project/wdawtiyguhtwdiqgnnmf
2. Click on **Authentication** in the left sidebar
3. Click on **Providers** tab
4. Scroll down to **Email** provider
5. **Uncheck** "Confirm email"
6. Click **Save**

Now users will be automatically confirmed without needing email verification!

### Option 2: Wait for Rate Limit to Reset

If you don't want to disable email confirmation:
- Wait 1-2 hours for the rate limit to reset
- Use different email addresses for testing
- Consider upgrading to a paid Supabase plan for higher limits

## For Production

> [!WARNING]
> **Before deploying to production**, you should:
> 1. Re-enable email confirmation for security
> 2. Set up a custom SMTP provider (like SendGrid, Mailgun, etc.)
> 3. Configure proper email templates

## Testing After Fix

1. After disabling email confirmation in Supabase
2. Go to http://localhost:5173/signup
3. Register a new user
4. You should see success immediately without email verification!
5. You can login right away

## Alternative: Use Different Emails

If you want to keep email confirmation enabled, use these temporary email services for testing:
- https://temp-mail.org
- https://10minutemail.com
- Use Gmail with the `+` trick: `youremail+test1@gmail.com`, `youremail+test2@gmail.com`
