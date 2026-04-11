-- RealWorld (Conduit) Supabase Seed
-- This script creates the necessary tables to support the Conduit data model.

-- 1. Create Profiles table (extension of users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
    username TEXT UNIQUE NOT NULL,
    bio TEXT,
    image TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Create Articles table
CREATE TABLE IF NOT EXISTS public.articles (
    id BIGSERIAL PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    body TEXT NOT NULL,
    author_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create Tags table
CREATE TABLE IF NOT EXISTS public.tags (
    id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

-- 4. Create ArticleTags junction table
CREATE TABLE IF NOT EXISTS public.article_tags (
    article_id BIGINT REFERENCES public.articles(id) ON DELETE CASCADE,
    tag_id BIGINT REFERENCES public.tags(id) ON DELETE CASCADE,
    PRIMARY KEY (article_id, tag_id)
);

-- 5. Create Comments table
CREATE TABLE IF NOT EXISTS public.comments (
    id BIGSERIAL PRIMARY KEY,
    body TEXT NOT NULL,
    article_id BIGINT REFERENCES public.articles(id) ON DELETE CASCADE,
    author_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Create Favorites table
CREATE TABLE IF NOT EXISTS public.favorites (
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    article_id BIGINT REFERENCES public.articles(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, article_id)
);

-- 7. Create Followers table
CREATE TABLE IF NOT EXISTS public.followers (
    follower_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    following_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    PRIMARY KEY (follower_id, following_id)
);

-- Enable RLS for all new tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.article_tags ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.favorites ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.followers ENABLE ROW LEVEL SECURITY;

-- Note: Add specific policies as needed for your application.
