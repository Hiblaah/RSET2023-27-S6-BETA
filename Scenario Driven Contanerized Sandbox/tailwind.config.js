/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'brand-dark': '#030303',
                'accent-purple': '#7c3aed',
                'accent-magenta': '#d946ef',
            }
        },
    },
    plugins: [],
}
