import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ProtectedRoute = ({ children }) => {
    const { user, isConfigured } = useAuth();

    // If Supabase is configured and there's no user, redirect to login
    if (isConfigured && !user) {
        return <Navigate to="/login" replace />;
    }

    // Otherwise, allow access (either user is logged in, or we are in Dev Bypass mode)
    return children;
};

export default ProtectedRoute;
