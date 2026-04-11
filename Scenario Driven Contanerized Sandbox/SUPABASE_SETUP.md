# Supabase Database Setup Instructions

## Step 1: Access Supabase SQL Editor

1. Go to your Supabase project dashboard: https://app.supabase.com
2. Select your project
3. Click on **SQL Editor** in the left sidebar

## Step 2: Run the Database Setup Script

1. Open the file `supabase-setup.sql` in this project
2. Copy all the SQL code
3. Paste it into the Supabase SQL Editor
4. Click **Run** to execute the script

## What This Script Does

- **Creates the `users` table** with all required fields:
  - `id` (UUID, linked to auth.users)
  - `email` (unique, required)
  - `username` (unique, required)
  - `full_name` (required)
  - `phone_number` (optional)
  - `date_of_birth` (optional)
  - `user_type` (optional)
  - `created_at` and `updated_at` (timestamps)

- **Enables Row Level Security (RLS)** to protect user data
- **Creates security policies** so users can only access their own data
- **Adds indexes** for faster username and email lookups
- **Creates triggers** to automatically update the `updated_at` timestamp

## Step 3: Verify the Setup

After running the script, verify the table was created:

1. Click on **Table Editor** in the left sidebar
2. You should see a `users` table
3. Click on it to view the schema

## Step 4: Start the Application

Once the database is set up, you can start using the application:

```bash
npm run dev
```

The authentication system is now fully functional!

## Troubleshooting

If you encounter any errors:

1. Make sure you're running the SQL in the correct project
2. Check that you have the necessary permissions
3. If the table already exists, you may need to drop it first (be careful with this in production!)
