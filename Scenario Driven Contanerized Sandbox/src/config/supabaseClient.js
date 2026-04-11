import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

// Check if credentials are valid (not empty and not placeholders)
const isConfigValid =
    supabaseUrl &&
    supabaseAnonKey &&
    supabaseUrl !== 'your_supabase_project_url' &&
    supabaseAnonKey !== 'your_supabase_anon_key' &&
    !supabaseAnonKey.toLowerCase().includes('mock') &&
    !supabaseAnonKey.toLowerCase().includes('placeholder') &&
    !supabaseAnonKey.startsWith('sb_publishable_');

if (!isConfigValid) {
    console.warn('Supabase credentials are missing or placeholder. Authentication features will be disabled.');
}

export const supabase = isConfigValid
    ? createClient(supabaseUrl, supabaseAnonKey)
    : null;
