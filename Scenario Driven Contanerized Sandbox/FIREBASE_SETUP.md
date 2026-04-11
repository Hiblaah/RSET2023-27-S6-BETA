# Firebase Setup Instructions

To get the authentication working, you need to create a project in Firebase and connect this project to it using configuration variables. Follow the steps below.

## 1. Create a Firebase Project
1. Go to the [Firebase Console](https://console.firebase.google.com/).
2. Click **Add project** and follow the prompts. (You do not need Google Analytics for this project, you can disable it).
3. Once the project is ready, click on the **Web** icon (</>) to add a web application to the project.
4. Name the app whatever you like, e.g., "CodeSandbox Clone", and click **Register App**.
5. You'll be presented with a configuration object that looks like this:

```javascript
const firebaseConfig = {
  apiKey: "AIzaSyXYZ...",
  authDomain: "your-project-id.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project-id.appspot.com",
  messagingSenderId: "1234567890",
  appId: "1:1234567890:web:abcd1234efgh"
};
```

Keep this tab open, you will need to copy these values.

## 2. Set Up Environment Variables
1. Within the `Sandbox_prj` folder, create a file named `.env`.
2. Add the following variables to `.env` using your Firebase config values from above:

```
VITE_FIREBASE_API_KEY=your_api_key_here
VITE_FIREBASE_AUTH_DOMAIN=your_auth_domain_here
VITE_FIREBASE_PROJECT_ID=your_project_id_here
VITE_FIREBASE_STORAGE_BUCKET=your_storage_bucket_here
VITE_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id_here
VITE_FIREBASE_APP_ID=your_app_id_here
```

## 3. Enable Authentication
1. Go back to the Firebase Console side menu and click **Build** -> **Authentication**.
2. Click **Get Started**.
3. Under the **Sign-in method** tab, click **Email/Password**.
4. Enable the first toggle switch for "Email/Password" and click **Save**.

## 4. Setup Firestore Database
1. Go back to the Firebase Console side menu and click **Build** -> **Firestore Database**.
2. Click **Create Database**.
3. Choose **Start in test mode** for local development (this allows open read/writes for 30 days) and click **Next**.
4. Choose an appropriate location for your database and click **Enable**.

## 5. You're set!
Once these steps are complete, run your development environment using `npm run dev` and your authentication system should be fully functional using Firebase!
