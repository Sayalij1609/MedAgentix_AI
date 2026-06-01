import { UserRole } from '../../../context/auth-context';

export interface LoginPayload {
  email: string;
  password?: string;
}

export interface RegisterPayload {
  name: string;
  email: string;
  password?: string;
  confirmPassword?: string;
  role: UserRole;
}

export interface AuthResponse {
  access_token: string;
  user: {
    user_id: string;
    email: string;
    role: UserRole;
    name: string;
    profile_id?: number;
  };
}

export type FormErrors<T> = {
  [K in keyof T]?: string;
} & {
  form?: string;
};
