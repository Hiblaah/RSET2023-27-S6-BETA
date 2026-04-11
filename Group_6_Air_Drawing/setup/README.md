# Project Setup Guide

This folder contains scripts and instructions to help you set up the Air Drawing project on another PC (Windows or macOS).

## Prerequisites

1.  **Node.js**: You must have Node.js installed on the target PC. You can download it from [nodejs.org](https://nodejs.org/).
2.  **Git**: Recommended for cloning the repository, but optional if you copy the files manually.

## Installation Steps

1.  **Copy the project files**: Copy the entire `air-drawing` folder to the new PC.

2.  **Install Dependencies**:

    *   **Windows**:
        *   Navigate to the `setup` folder.
        *   Double-click `install.bat`.

    *   **macOS / Linux**:
        *   Open Terminal.
        *   Navigate to the `setup` folder: `cd path/to/air-drawing/setup`
        *   Make the script executable: `chmod +x install.sh`
        *   Run the script: `./install.sh`

3.  **Environment Variables**:
    *   In the root folder of the project (where `package.json` is), locate the file named `.env.example`.
    *   Create a copy of this file and rename it to `.env` (note the dot at the beginning).
    *   Open `.env` in a text editor and fill in your Supabase URL and Anon Key:
        ```
        VITE_SUPABASE_URL=your_supabase_url_here
        VITE_SUPABASE_ANON_KEY=your_supabase_anon_key_here
        ```
    *   *Note: If you don't have these keys, ask the project owner or check your Supabase dashboard.*

## Running the App

After installation, you can run the app by opening a terminal in the project folder and running:

```bash
npm run dev
```
