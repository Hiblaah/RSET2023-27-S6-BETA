package com.phonecluster.app.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Storage
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.phonecluster.app.storage.PreferencesManager
import com.phonecluster.app.utils.DeviceInfoProvider
import androidx.compose.foundation.BorderStroke

// ─── Color Tokens ─────────────────────────────────────────────────────────────

private val BgDeep        = Color(0xFF020617)
private val BgCard        = Color(0xFF0D1424)
private val BgCardAlt     = Color(0xFF0A1120)
private val BorderSubtle  = Color(0xFF1E293B)
private val AccentCyan    = Color(0xFF22D3EE)
private val AccentPurple  = Color(0xFFA78BFA)
private val AccentGreen   = Color(0xFF34D399)
private val AccentAmber   = Color(0xFFFBBF24)
private val TextPrimary   = Color(0xFFF1F5F9)
private val TextSecondary = Color(0xFF94A3B8)
private val TextMuted     = Color(0xFF475569)
private val ErrorRed      = Color(0xFFEF4444)

// ─── Screen ───────────────────────────────────────────────────────────────────

@Composable
fun StorageModeScreen(
    onBackClick: (() -> Unit)? = null
) {
    val context  = LocalContext.current
    val deviceId = PreferencesManager.getDeviceId(context) ?: -1

    android.util.Log.d(
        "FINGERPRINT",
        DeviceInfoProvider.getDeviceFingerprint(context)
    )

    // ── Pulse animations ──────────────────────────────────────────────────────
    val inf = rememberInfiniteTransition(label = "node")
    val p1 by inf.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(2600, easing = LinearEasing)),
        label = "p1"
    )
    val p2 by inf.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(2600, easing = LinearEasing, delayMillis = 867)),
        label = "p2"
    )
    val p3 by inf.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(2600, easing = LinearEasing, delayMillis = 1734)),
        label = "p3"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Spacer(modifier = Modifier.height(52.dp))

            // ── Node visualization ────────────────────────────────────────────
            Box(
                modifier = Modifier.size(240.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val c = this.center
                    val pulses = listOf(p1, p2, p3)

                    // Outermost static dashed ring
                    drawCircle(
                        color = AccentCyan,
                        radius = size.minDimension / 2f * 0.92f,
                        center = c,
                        alpha = 0.08f,
                        style = Stroke(width = 1.dp.toPx())
                    )
                    drawCircle(
                        color = AccentCyan,
                        radius = size.minDimension / 2f * 0.72f,
                        center = c,
                        alpha = 0.12f,
                        style = Stroke(width = 1.dp.toPx())
                    )

                    // Travelling pulse rings
                    pulses.forEach { p ->
                        val ringR = 54.dp.toPx() + p * 58.dp.toPx()
                        val ringA = (1f - p) * 0.35f
                        drawCircle(
                            color = AccentCyan,
                            radius = ringR,
                            center = c,
                            alpha = ringA,
                            style = Stroke(width = 1.5.dp.toPx())
                        )
                    }
                }

                // ── Center node circle ────────────────────────────────────────
                Box(
                    modifier = Modifier
                        .size(108.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(AccentCyan.copy(0.22f), BgCard),
                                radius = 200f
                            )
                        )
                        .border(1.5.dp, AccentCyan.copy(0.45f), CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(3.dp)
                    ) {
                        Text(
                            text = "#$deviceId",
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Bold,
                            color = AccentCyan,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text = "NODE",
                            fontSize = 9.sp,
                            color = AccentCyan.copy(0.7f),
                            letterSpacing = 2.5.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // ── Active badge ──────────────────────────────────────────────────
            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(AccentGreen.copy(alpha = 0.1f))
                    .border(1.dp, AccentGreen.copy(alpha = 0.3f), RoundedCornerShape(20.dp))
                    .padding(horizontal = 14.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(AccentGreen)
                )
                Text(
                    text = "Storage Node Active",
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium,
                    color = AccentGreen
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = "Serving as a storage node in the PocketCluster network",
                fontSize = 12.sp,
                color = TextMuted,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(28.dp))

            // ── Stats row ─────────────────────────────────────────────────────
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                NodeStatBox(
                    label      = "Status",
                    value      = "ONLINE",
                    valueColor = AccentGreen,
                    modifier   = Modifier.weight(1f)
                )
                NodeStatBox(
                    label      = "Mode",
                    value      = "STORAGE",
                    valueColor = AccentPurple,
                    modifier   = Modifier.weight(1f)
                )
                NodeStatBox(
                    label      = "Network",
                    value      = "P2P",
                    valueColor = AccentCyan,
                    modifier   = Modifier.weight(1f)
                )
            }

            Spacer(modifier = Modifier.height(10.dp))

            // ── Activity card ─────────────────────────────────────────────────
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = BgCard),
                elevation = CardDefaults.cardElevation(0.dp),
                border = BorderStroke(1.dp, BorderSubtle)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {

                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(30.dp)
                                .clip(RoundedCornerShape(8.dp))
                                .background(AccentCyan.copy(0.1f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Outlined.Storage,
                                contentDescription = null,
                                tint = AccentCyan,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                        Text(
                            text = "Activity",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary
                        )
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    ActivityRow(
                        dotColor = AccentAmber,
                        text     = "Waiting for chunk assignments..."
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    ActivityRow(
                        dotColor = AccentGreen,
                        text     = "Heartbeat: connected to coordinator"
                    )
                }
            }
        }
    }
}

// ─── Node Stat Box ────────────────────────────────────────────────────────────

@Composable
private fun NodeStatBox(
    label: String,
    value: String,
    valueColor: Color,
    modifier: Modifier = Modifier
) {
    Card(
        modifier  = modifier,
        shape     = RoundedCornerShape(14.dp),
        colors    = CardDefaults.cardColors(containerColor = BgCard),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = BorderStroke(1.dp, BorderSubtle)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(5.dp)
        ) {
            Text(
                text        = value,
                fontSize    = 13.sp,
                fontWeight  = FontWeight.Bold,
                color       = valueColor,
                fontFamily  = FontFamily.Monospace,
                letterSpacing = 1.sp
            )
            Text(
                text       = label,
                fontSize   = 10.sp,
                color      = TextMuted,
                letterSpacing = 0.5.sp
            )
        }
    }
}

// ─── Activity Row ─────────────────────────────────────────────────────────────

@Composable
private fun ActivityRow(dotColor: Color, text: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Box(
            modifier = Modifier
                .size(7.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Text(
            text     = text,
            fontSize = 13.sp,
            color    = TextSecondary
        )
    }
}