# CodeSandbox - Premium Scenario IDE 🚀

A high-end, browser-based coding platform built for the next generation of developers. Featuring a sophisticated **Monaco Engine IDE**, interactive coding scenarios, and a stunning **Dark Neon aesthetic**.

![CodeSandbox Preview](file:///C:/Users/HP/.gemini/antigravity/brain/d5925da6-891c-45c5-a689-261c6c748479/uploaded_media_1770229219006.png)

## ✨ Premium Features

- **🎨 Sophisticated Aesthetics**: A curated dark aesthetic with vibrant purple/magenta light streaks, glassmorphism, and neon accents.
- **💻 Interactive IDE Scenarios**: Full-featured code editor using the **Monaco Engine** (the power behind VS Code) with integrated test suites.
- **📊 Real-Time Progress Tracking**: Dynamic dashboard that tracks submissions and calculates average scores in real-time.
- **🔐 Enhanced Security**: Secure authentication flow with **password visibility toggles** and robust error handling.
- **🛠️ Development Mode Bypass**: Accelerated UI review cycle—skip authentication to jump straight into the dashboard and IDE.
- **📱 Ultra-Responsive**: Designed to look stunning on every screen size, from desktops to tablets.

## 🛠️ Tech Stack

- **Frontend**: [React 18](https://reactjs.org/)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) with custom glassmorphism utilities.
- **Editor Engine**: [@monaco-editor/react](https://github.com/suren-atoyan/monaco-react)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Backend**: [Supabase](https://supabase.com/) (Auth & Postgres)

## 📦 Quick Start

### 1. Installation
```bash
git clone [repository-url]
cd Sandbox_prj
npm install
```

### 2. Environment Setup
Rename `.env.example` to `.env` and add your Supabase credentials:
```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. Launch Development Server
```bash
npm run dev
```
> [!TIP]
> **Development Mode Bypass**: If Supabase is not yet configured, a blue "Skip to Dashboard" button will appear on the login screen, allowing you to review the IDE and Dashboard UI immediately.

## 🏗️ Project Structure

- `src/components/`: Reusable UI modules (Cards, Nav, etc.)
- `src/pages/`: Core application views (Login, SignUp, Dashboard, Editor)
- `src/context/`: State management (Auth & User Profile)
- `src/config/`: Service initializations (Supabase Client)
- `src/index.css`: Global design system and premium tokens

## 📜 Available Scripts

- `npm run dev`: Starts the ultra-fast Vite dev server.
- `npm run build`: Optimizes for production deployment.
- `npm run lint`: Ensures code quality and standard compliance.

---

Built with ❤️ for Developers.
