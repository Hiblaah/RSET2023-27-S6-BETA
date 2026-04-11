package com.phonecluster.app.screens

import android.content.Intent
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.phonecluster.app.storage.PreferencesManager
import com.phonecluster.app.storage.StorageNodeService

// ─── Color Tokens ─────────────────────────────────────────────────────────────

private val BgDeep        = Color(0xFF020617)
private val BgCard        = Color(0xFF0D1424)
private val BorderSubtle  = Color(0xFF1E293B)
private val AccentCyan    = Color(0xFF22D3EE)
private val AccentPurple  = Color(0xFFA78BFA)
private val AccentGreen   = Color(0xFF34D399)
private val TextPrimary   = Color(0xFFF1F5F9)
private val TextSecondary = Color(0xFF94A3B8)
private val TextMuted     = Color(0xFF475569)

// ─── Screen ───────────────────────────────────────────────────────────────────

@Composable
fun ModeSelectionScreen(
    onUserModeClick: () -> Unit = {},
    onStorageModeClick: () -> Unit = {}
) {
    val context  = LocalContext.current
    val deviceId = PreferencesManager.getDeviceId(context) ?: -1

    val pulse = rememberInfiniteTransition(label = "pulse")
    val glowAlpha by pulse.animateFloat(
        initialValue  = 0.08f,
        targetValue   = 0.18f,
        animationSpec = infiniteRepeatable(
            animation  = tween(3000, easing = EaseInOutSine),
            repeatMode = RepeatMode.Reverse
        ),
        label = "glowAlpha"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
    ) {

        // ── Subtle grid background ────────────────────────────────────────────
        Canvas(modifier = Modifier.fillMaxSize()) {
            val gridSize = 48.dp.toPx()
            val lineAlpha = 0.035f

            var x = 0f
            while (x <= size.width) {
                drawLine(
                    color       = AccentCyan,
                    start       = Offset(x, 0f),
                    end         = Offset(x, size.height),
                    alpha       = lineAlpha,
                    strokeWidth = 1f
                )
                x += gridSize
            }
            var y = 0f
            while (y <= size.height) {
                drawLine(
                    color       = AccentCyan,
                    start       = Offset(0f, y),
                    end         = Offset(size.width, y),
                    alpha       = lineAlpha,
                    strokeWidth = 1f
                )
                y += gridSize
            }
        }

        // ── Ambient radial glow ───────────────────────────────────────────────
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(380.dp)
                .align(Alignment.TopCenter)
                .drawBehind {
                    drawCircle(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                AccentCyan.copy(alpha = glowAlpha),
                                Color.Transparent
                            ),
                            center = Offset(size.width / 2f, size.height * 0.4f),
                            radius = size.width * 0.72f
                        )
                    )
                }
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp),
            verticalArrangement   = Arrangement.Center,
            horizontalAlignment   = Alignment.CenterHorizontally
        ) {

            HeroSection()

            Spacer(modifier = Modifier.height(36.dp))

            DeviceIdBadge(deviceId = deviceId)

            Spacer(modifier = Modifier.height(36.dp))

            ModeCard(
                title       = "User Mode",
                subtitle    = "Upload, browse & manage your files",
                icon        = Icons.Outlined.CloudUpload,
                accentColor = AccentCyan,
                bgGradient  = Brush.horizontalGradient(
                    listOf(AccentCyan.copy(alpha = 0.09f), Color.Transparent)
                ),
                onClick = onUserModeClick
            )

            Spacer(modifier = Modifier.height(12.dp))

            ModeCard(
                title       = "Storage Mode",
                subtitle    = "Contribute storage space to the cluster",
                icon        = Icons.Outlined.Storage,
                accentColor = AccentPurple,
                bgGradient  = Brush.horizontalGradient(
                    listOf(AccentPurple.copy(alpha = 0.09f), Color.Transparent)
                ),
                onClick = {
                    if (deviceId != -1) {
                        val intent = Intent(context, StorageNodeService::class.java)
                        intent.putExtra("deviceId", deviceId)
                        ContextCompat.startForegroundService(context, intent)
                    }
                    onStorageModeClick()
                }
            )

            Spacer(modifier = Modifier.height(32.dp))

            Row(
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(
                    imageVector    = Icons.Outlined.Info,
                    contentDescription = null,
                    tint           = TextMuted,
                    modifier       = Modifier.size(12.dp)
                )
                Text(
                    text      = "You can switch modes anytime from settings",
                    fontSize  = 12.sp,
                    color     = TextMuted,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

// ─── Hero Section ─────────────────────────────────────────────────────────────

@Composable
private fun HeroSection() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Layered icon cluster
        Box(contentAlignment = Alignment.Center) {
            // Outer glow ring
            Box(
                modifier = Modifier
                    .size(96.dp)
                    .clip(RoundedCornerShape(28.dp))
                    .background(
                        Brush.radialGradient(
                            listOf(AccentCyan.copy(0.08f), Color.Transparent)
                        )
                    )
            )

            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(22.dp))
                    .background(
                        Brush.linearGradient(
                            listOf(AccentCyan.copy(0.18f), AccentPurple.copy(0.12f))
                        )
                    )
                    .border(
                        width = 1.dp,
                        brush = Brush.linearGradient(
                            listOf(AccentCyan.copy(0.5f), AccentPurple.copy(0.3f))
                        ),
                        shape = RoundedCornerShape(22.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector    = Icons.Outlined.Hub,
                    contentDescription = null,
                    tint           = AccentCyan,
                    modifier       = Modifier.size(38.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(2.dp))

        Text(
            text          = "PocketCluster",
            fontSize      = 30.sp,
            fontWeight    = FontWeight.Bold,
            color         = TextPrimary,
            letterSpacing = (-0.5).sp
        )

        Text(
            text          = "Decentralized personal storage",
            fontSize      = 13.sp,
            color         = TextMuted,
            letterSpacing = 0.2.sp
        )
    }
}

// ─── Device ID Badge ──────────────────────────────────────────────────────────

@Composable
private fun DeviceIdBadge(deviceId: Int) {
    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(Color(0xFF0A1628))
            .border(1.dp, AccentCyan.copy(0.25f), RoundedCornerShape(20.dp))
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Pulsing active dot
        Box(
            modifier = Modifier
                .size(6.dp)
                .clip(CircleShape)
                .background(AccentGreen)
        )

        Text(
            text          = "Device",
            fontSize      = 12.sp,
            color         = TextMuted,
            letterSpacing = 0.3.sp
        )

        Text(
            text          = "#$deviceId",
            fontSize      = 13.sp,
            fontWeight    = FontWeight.Bold,
            color         = AccentCyan,
            fontFamily    = FontFamily.Monospace
        )
    }
}

// ─── Mode Card ────────────────────────────────────────────────────────────────

@Composable
private fun ModeCard(
    title: String,
    subtitle: String,
    icon: ImageVector,
    accentColor: Color,
    bgGradient: Brush,
    onClick: () -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()

    val scale by animateFloatAsState(
        targetValue   = if (isPressed) 0.96f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness    = Spring.StiffnessHigh
        ),
        label = "cardScale"
    )

    val borderAlpha by animateFloatAsState(
        targetValue   = if (isPressed) 0.8f else 0.35f,
        animationSpec = tween(100),
        label         = "borderAlpha"
    )

    val iconBgAlpha by animateFloatAsState(
        targetValue   = if (isPressed) 0.22f else 0.10f,
        animationSpec = tween(100),
        label         = "iconBgAlpha"
    )

    val arrowOffset by animateFloatAsState(
        targetValue   = if (isPressed) 5f else 0f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness    = Spring.StiffnessMedium
        ),
        label = "arrowOffset"
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .clickable(
                interactionSource = interactionSource,
                indication        = null
            ) { onClick() },
        shape     = RoundedCornerShape(18.dp),
        colors    = CardDefaults.cardColors(containerColor = BgCard),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = androidx.compose.foundation.BorderStroke(
            width = 1.dp,
            brush = Brush.horizontalGradient(
                listOf(
                    accentColor.copy(alpha = borderAlpha),
                    BorderSubtle,
                    BorderSubtle
                )
            )
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(bgGradient)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp, vertical = 20.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Icon box
                Box(
                    modifier = Modifier
                        .size(52.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(accentColor.copy(alpha = iconBgAlpha))
                        .border(
                            1.dp,
                            accentColor.copy(alpha = borderAlpha * 0.5f),
                            RoundedCornerShape(14.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector    = icon,
                        contentDescription = title,
                        tint           = accentColor,
                        modifier       = Modifier.size(26.dp)
                    )
                }

                // Text
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text          = title,
                        fontSize      = 16.sp,
                        fontWeight    = FontWeight.SemiBold,
                        color         = TextPrimary,
                        letterSpacing = (-0.2).sp
                    )
                    Spacer(modifier = Modifier.height(3.dp))
                    Text(
                        text     = subtitle,
                        fontSize = 12.sp,
                        color    = TextSecondary
                    )
                }

                // Arrow
                Icon(
                    imageVector    = Icons.Default.ChevronRight,
                    contentDescription = null,
                    tint           = accentColor.copy(alpha = if (isPressed) 1f else 0.6f),
                    modifier       = Modifier
                        .size(20.dp)
                        .graphicsLayer { translationX = arrowOffset }
                )
            }
        }
    }
}