package com.phonecluster.app.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.Color

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF22D3EE),
    onPrimary = Color.Black,

    background = Color(0xFF020617),
    onBackground = Color(0xFFF8FAFC),

    surface = Color(0xFF111827),
    onSurface = Color(0xFFF8FAFC),

    surfaceVariant = Color(0xFF1E293B),
    onSurfaceVariant = Color(0xFFCBD5F5),

    secondary = Color(0xFF06B6D4),
    onSecondary = Color.Black,

    outline = Color(0xFF334155)
)

private val LightColorScheme = DarkColorScheme

@Composable
fun CloudStorageAppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // Dynamic color is available on Android 12+
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }

        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        content = content
    )
}