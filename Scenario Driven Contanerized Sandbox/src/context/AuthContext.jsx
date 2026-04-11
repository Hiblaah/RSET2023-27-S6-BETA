import React, { createContext, useContext, useState, useEffect } from 'react';
import { auth, db, isConfigValid } from '../config/firebaseClient';
import { 
    createUserWithEmailAndPassword, 
    signInWithEmailAndPassword, 
    signOut as firebaseSignOut,
    onAuthStateChanged 
} from 'firebase/auth';
import { doc, setDoc, getDoc, collection, query, where, getDocs } from 'firebase/firestore';

const AuthContext = createContext({});

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!isConfigValid || !auth) {
            setLoading(false);
            return;
        }

        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
            setLoading(false);
        });

        return () => unsubscribe();
    }, []);

    // Sign up function
    const signUp = async (email, password, userData) => {
        if (!isConfigValid || !auth) {
            return { data: null, error: { message: 'Firebase is not configured. Please check your .env file.' } };
        }
        try {
            // Check if username already exists
            const usersRef = collection(db, 'users');
            const q = query(usersRef, where('username', '==', userData.username));
            const querySnapshot = await getDocs(q);
            
            if (!querySnapshot.empty) {
                throw new Error('Username is already taken.');
            }

            const userCredential = await createUserWithEmailAndPassword(auth, email, password);
            const user = userCredential.user;

            // Prepare user profile object, removing undefined values
            const userProfile = {
                id: user.uid,
                email: email,
                username: userData.username,
                full_name: userData.full_name || null,
                phone_number: userData.phoneNumber || null,
                date_of_birth: userData.dateOfBirth || null,
                user_type: userData.userType || null,
                createdAt: new Date().toISOString()
            };

            // Remove any remaining undefined fields to prevent Firestore errors
            Object.keys(userProfile).forEach(key => userProfile[key] === undefined && delete userProfile[key]);

            // Insert additional user data into Firestore
            await setDoc(doc(db, 'users', user.uid), userProfile);

            return { data: user, error: null };
        } catch (error) {
            // Handle specific Firebase auth errors if needed
            return { data: null, error };
        }
    };

    // Sign in function
    const signIn = async (username, password) => {
        if (!isConfigValid || !auth) {
            return { data: null, error: { message: 'Firebase is not configured. Please check your .env file.' } };
        }
        try {
            // First, get the email associated with the username
            const usersRef = collection(db, 'users');
            const q = query(usersRef, where('username', '==', username));
            const querySnapshot = await getDocs(q);

            if (querySnapshot.empty) {
                throw new Error('Invalid username or password');
            }

            let email = null;
            querySnapshot.forEach((doc) => {
                email = doc.data().email;
            });

            if (!email) {
                throw new Error('Invalid username or password');
            }

            // Then sign in with email and password
            const userCredential = await signInWithEmailAndPassword(auth, email, password);

            return { data: userCredential.user, error: null };
        } catch (error) {
            return { data: null, error };
        }
    };

    // Sign out function
    const signOut = async () => {
        if (!auth) return { error: null };
        try {
            await firebaseSignOut(auth);
            localStorage.clear();
            return { error: null };
        } catch (error) {
            return { error };
        }
    };

    const value = {
        user,
        loading,
        signUp,
        signIn,
        signOut,
        isConfigured: isConfigValid,
    };

    return (
        <AuthContext.Provider value={value}>
            {loading ? (
                <div className="min-h-screen bg-slate-900 flex items-center justify-center">
                    <div className="flex flex-col items-center gap-4">
                        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-slate-400 font-medium">Loading CodeSandbox...</p>
                    </div>
                </div>
            ) : children}
        </AuthContext.Provider>
    );
};
