// MedAgentix AI — Enterprise Design System Theme Constants
// Aligned with the professional healthcare SaaS specification

export const THEME_COLORS = {
  primary: {
    DEFAULT: '#0F4C81', // Clinical Deep Blue (authority, trust)
    light: '#1e68a3',
    dark: '#0a3254',
  },
  secondary: {
    DEFAULT: '#14B8A6', // Calm Clinical Teal (care, hygiene, health)
    light: '#2dd4bf',
    dark: '#0f766e',
  },
  background: {
    light: '#F8FAFC',   // Off-white slate background for minimal eye strain
    dark: '#0F172A',    // Deep slate blue dark background
  },
  success: {
    DEFAULT: '#10B981', // Medical safe green (normal values)
    light: '#34d399',
    dark: '#047857',
  },
  warning: {
    DEFAULT: '#F59E0B', // Caution amber yellow (borderline/attention values)
    light: '#fbbf24',
    dark: '#b45309',
  },
  danger: {
    DEFAULT: '#EF4444', // Alert red (critical values/emergencies)
    light: '#f87171',
    dark: '#b91c1c',
  },
  neutral: {
    slate: {
      50: '#F8FAFC',
      100: '#F1F5F9',
      200: '#E2E8F0',
      300: '#CBD5E1',
      400: '#94A3B8',
      500: '#64748B',
      600: '#475569',
      700: '#334155',
      800: '#1E293B',
      900: '#0F172A',
    }
  }
} as const;

export const SHADOWS = {
  sm: '0 1px 2px 0 rgba(15, 76, 129, 0.05)',
  md: '0 4px 6px -1px rgba(15, 76, 129, 0.08), 0 2px 4px -2px rgba(15, 76, 129, 0.04)',
  lg: '0 10px 15px -3px rgba(15, 76, 129, 0.1), 0 4px 6px -4px rgba(15, 76, 129, 0.05)',
  glow: '0 0 15px 2px rgba(20, 184, 166, 0.15)', // Custom secondary accent glow
} as const;

export const TYPOGRAPHY = {
  fontFamily: "'Inter', 'Outfit', sans-serif",
  scales: {
    xs: '0.75rem',     // Caption/Details
    sm: '0.875rem',    // Standard body text / form labels
    base: '1rem',      // Primary readability text
    lg: '1.125rem',    // Subheaders
    xl: '1.25rem',     // Section titles
    '2xl': '1.5rem',   // Panel sub-titles
    '3xl': '1.875rem', // Primary dashboard headings
    '4xl': '2.25rem',  // Main hero statements
  }
} as const;
