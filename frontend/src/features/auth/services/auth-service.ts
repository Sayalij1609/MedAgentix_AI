import apiClient from '../../../services/api-client';
import { LoginPayload, RegisterPayload, AuthResponse } from '../types';

export const AuthService = {
  /**
   * Authenticates a user with the backend API.
   * Maps response attributes to match the frontend session structure.
   */
  async login(payload: LoginPayload): Promise<AuthResponse> {
    try {
      const response = await apiClient.post('/auth/login', {
        email: payload.email,
        password: payload.password,
      });

      const { access_token, user } = response.data;

      return {
        access_token,
        user: {
          user_id: String(user.id),
          email: user.email,
          role: user.role, // matches lowercase standardized role ('patient' | 'doctor' | 'admin')
          name: user.name,
          // Assign mock profile IDs if profile relation data is not returned yet
          profile_id: user.role === 'doctor' ? 101 : 202,
        },
      };
    } catch (error: any) {
      const message = error.response?.data?.message || 'Invalid clinical credentials. Please verify your email and password.';
      throw new Error(message);
    }
  },

  /**
   * Registers a new user.
   * Automatically invokes login on success to obtain the JWT token and maintain
   * the auto-login user experience.
   */
  async register(payload: RegisterPayload): Promise<AuthResponse> {
    try {
      // 1. Trigger registration request
      await apiClient.post('/auth/register', {
        name: payload.name,
        email: payload.email,
        password: payload.password,
        role: payload.role,
      });

      // 2. Perform automatic login to fetch session token and payload
      return await this.login({
        email: payload.email,
        password: payload.password,
      });
    } catch (error: any) {
      const message = error.response?.data?.message || 'An unexpected registration error occurred.';
      throw new Error(message);
    }
  },

  /**
   * Fetches the profile details of the authenticated user.
   */
  async getProfile(): Promise<{ success: boolean; user: any }> {
    try {
      const response = await apiClient.get('/auth/profile');
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.message || 'Failed to retrieve profile.';
      throw new Error(message);
    }
  },

  /**
   * Updates the profile details of the authenticated user.
   */
  async updateProfile(payload: { name?: string; email?: string; password?: string }): Promise<{ success: boolean; message: string; user: any }> {
    try {
      const response = await apiClient.put('/auth/profile', payload);
      return response.data;
    } catch (error: any) {
      const message = error.response?.data?.message || 'Failed to update profile details.';
      throw new Error(message);
    }
  },
};
