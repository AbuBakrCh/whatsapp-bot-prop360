module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        whatsapp: {
          50: '#f6fff8',
          100: '#ecfff1',
          300: '#bff2c6',
          500: '#2dbc3a' // accent green
        }
      },
      fontFamily: {
        display: ['Fraunces', 'Georgia', 'serif'],
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        'welcome-rise': {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'welcome-fade': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      animation: {
        'welcome-rise': 'welcome-rise 0.55s ease-out both',
        'welcome-fade': 'welcome-fade 0.7s ease-out both',
      },
    }
  },
  plugins: []
}
