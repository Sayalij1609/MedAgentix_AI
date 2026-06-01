import { LoginPayload, RegisterPayload, AuthResponse } from '../types';

/**
 * Simulated authentication service mimicking REST API interactions
 * with artificial network latency and validation rules.
 */

// Helper to simulate network lag
const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const AuthService = {
  /**
   * Simulates a secure user login request.
   * Special mock test case: Logging in with "error@medagentix.ai" will simulate a credentials failure.
   */
  async login(payload: LoginPayload): Promise<AuthResponse> {
    await delay(1000); // 1s simulation

    const { email } = payload;

    // Simulate validation error scenario
    if (email.toLowerCase() === 'error@medagentix.ai') {
      throw new Error('Invalid clinical credentials. Please verify your email and password.');
    }

    // Determine role based on email hint, default to Patient
    let role: 'Patient' | 'Doctor' | 'Admin' = 'Patient';
    let name = 'Dr. Sarah Jenkins';
    
    if (email.toLowerCase().includes('doctor') || email.toLowerCase().includes('doc')) {
      role = 'Doctor';
      name = 'Dr. Sarah Jenkins';
    } else if (email.toLowerCase().includes('admin')) {
      role = 'Admin';
      name = 'Clinical Administrator';
    } else {
      name = 'Alexander Vance';
    }

    return {
      access_token: 'mock-jwt-token-xyz-123456789',
      user: {
        user_id: 'usr_mock_123',
        email: email,
        role: role,
        name: name,
        profile_id: role === 'Doctor' ? 101 : 202,
      },
    };
  },

  /**
   * Simulates a secure user registration request.
   * Special mock test case: Emailing "taken@medagentix.ai" simulates a database duplicate conflict.
   */
  async register(payload: RegisterPayload): Promise<AuthResponse> {
    await delay(1200); // 1.2s simulation

    const { name, email, role } = payload;

    if (email.toLowerCase() === 'taken@medagentix.ai') {
      throw new Error('An account with this clinical email address already exists.');
    }

    return {
      access_token: 'mock-jwt-token-abc-987654321',
      user: {
        user_id: 'usr_mock_456',
        email: email,
        role: role,
        name: name,
        profile_id: role === 'Doctor' ? 303 : 404,
      },
    };
  },
};
