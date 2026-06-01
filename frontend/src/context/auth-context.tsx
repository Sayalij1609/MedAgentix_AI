import React, { createContext, useContext, useState, useEffect } from 'react';

export type UserRole = 'Patient' | 'Doctor' | 'Admin';

export interface UserSession {
  user_id: string;
  email: string;
  role: UserRole;
  profile_id?: number; // Links to patients.id or doctors.id
}

interface AuthContextType {
  user: UserSession | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (token: string, user: UserSession) => void;
  logout: () => void;
  updateProfileId: (profileId: number) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserSession | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    // Re-hydrate session from local storage on bootstrap
    const savedToken = localStorage.getItem('access_token');
    const savedUser = localStorage.getItem('user_session');

    if (savedToken && savedUser) {
      try {
        setToken(savedToken);
        setUser(JSON.parse(savedUser));
      } catch (e) {
        console.error('Session hydration failed', e);
        logout();
      }
    }
    setIsLoading(false);
  }, []);

  const login = (accessToken: string, userSession: UserSession) => {
    setToken(accessToken);
    setUser(userSession);
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('user_session', JSON.stringify(userSession));
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_session');
  };

  const updateProfileId = (profileId: number) => {
    if (!user) return;
    const updatedUser = { ...user, profile_id: profileId };
    setUser(updatedUser);
    localStorage.setItem('user_session', JSON.stringify(updatedUser));
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!token,
        isLoading,
        login,
        logout,
        updateProfileId,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};
