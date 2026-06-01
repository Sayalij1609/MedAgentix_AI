import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth, UserRole } from '../context/auth-context';
import { ROUTES } from './config';

interface RoleRouteProps {
  children: React.ReactElement;
  allowedRoles: UserRole[];
}

export const RoleRoute: React.FC<RoleRouteProps> = ({
  children,
  allowedRoles,
}) => {
  const { user, isAuthenticated } = useAuth();

  if (!isAuthenticated || !user) {
    return <Navigate to={ROUTES.LOGIN} replace />;
  }

  if (!allowedRoles.includes(user.role)) {
    // Redirect role exceptions to their default home dashboard
    const defaultRedirect =
      user.role === 'Doctor' ? ROUTES.DOCTOR_DASHBOARD : ROUTES.PATIENT_DASHBOARD;
    return <Navigate to={defaultRedirect} replace />;
  }

  return children;
};
