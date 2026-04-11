package com.phonecluster.app.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Hub
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.phonecluster.app.network.ApiClient
import com.phonecluster.app.network.DeviceRegistrationRequest
import com.phonecluster.app.storage.PreferencesManager
import com.phonecluster.app.utils.DeviceInfoProvider
import kotlinx.coroutines.launch

// ─── Color Tokens ─────────────────────────────────────────────────────────────

private val BgDeep       = Color(0xFF020617)
private val BgCard       = Color(0xFF0D1424)
private val BorderSubtle = Color(0xFF1E293B)
private val AccentCyan   = Color(0xFF22D3EE)
private val AccentPurple = Color(0xFFA78BFA)
private val AccentGreen  = Color(0xFF34D399)
private val ErrorRed     = Color(0xFFEF4444)
private val TextPrimary  = Color(0xFFF1F5F9)
private val TextSecondary= Color(0xFF94A3B8)
private val TextMuted    = Color(0xFF475569)

// ─── Screen ───────────────────────────────────────────────────────────────────

@Composable
fun RegistrationScreen(onRegistered: () -> Unit) {
    val context = LocalContext.current
    val scope   = rememberCoroutineScope()

    var statusText by remember { mutableStateOf("not registered") }
    var isLoading  by remember { mutableStateOf(false) }
    var isError    by remember { mutableStateOf(false) }
    var isSuccess  by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        if (PreferencesManager.isRegistered(context)) {
            val deviceId = PreferencesManager.getDeviceId(context)
            statusText = "registered (device_id = $deviceId)"
            onRegistered()
        }
    }

    // ── Background node pulse animations ──────────────────────────────────────
    val inf = rememberInfiniteTransition(label = "bg")
    val p1 by inf.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(3000, easing = LinearEasing)),
        label = "p1"
    )
    val p2 by inf.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(3000, easing = LinearEasing, delayMillis = 1000)),
        label = "p2"
    )
    val p3 by inf.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(3000, easing = LinearEasing, delayMillis = 2000)),
        label = "p3"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep),
        contentAlignment = Alignment.Center
    ) {

        // ── Animated network background ───────────────────────────────────────
        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height

            val nodes = listOf(
                Offset(w * 0.12f, h * 0.15f),
                Offset(w * 0.85f, h * 0.20f),
                Offset(w * 0.06f, h * 0.70f),
                Offset(w * 0.90f, h * 0.68f),
                Offset(w * 0.50f, h * 0.06f),
                Offset(w * 0.30f, h * 0.88f),
                Offset(w * 0.72f, h * 0.90f),
            )

            // Connection lines
            val connections = listOf(
                0 to 1, 0 to 4, 1 to 4, 0 to 2,
                1 to 3, 2 to 3, 2 to 5, 3 to 6, 5 to 6
            )
            connections.forEach { (a, b) ->
                drawLine(
                    color = AccentCyan,
                    start = nodes[a],
                    end = nodes[b],
                    alpha = 0.07f,
                    strokeWidth = 1.dp.toPx()
                )
            }

            // Pulsing rings per node
            val pulses = listOf(p1, p2, p3, p1, p2, p3, p1)
            nodes.forEachIndexed { i, pos ->
                val pulse = pulses[i]
                val ringR = 12.dp.toPx() + pulse * 28.dp.toPx()
                val ringA = (1f - pulse) * 0.22f
                drawCircle(
                    color = if (i % 2 == 0) AccentCyan else AccentPurple,
                    radius = ringR,
                    center = pos,
                    alpha = ringA,
                    style = Stroke(width = 1.dp.toPx())
                )
                // Core dot
                drawCircle(
                    color = if (i % 2 == 0) AccentCyan else AccentPurple,
                    radius = 3.dp.toPx(),
                    center = pos,
                    alpha = 0.55f
                )
            }
        }

        // ── Centered content ──────────────────────────────────────────────────
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            // ── Logo / brand ──────────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .size(84.dp)
                    .clip(RoundedCornerShape(24.dp))
                    .background(
                        Brush.linearGradient(
                            listOf(AccentCyan.copy(0.18f), AccentPurple.copy(0.12f))
                        )
                    )
                    .border(
                        1.dp,
                        Brush.linearGradient(
                            listOf(AccentCyan.copy(0.5f), AccentPurple.copy(0.3f))
                        ),
                        RoundedCornerShape(24.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Outlined.Hub,
                    contentDescription = null,
                    tint = AccentCyan,
                    modifier = Modifier.size(40.dp)
                )
            }

            Spacer(modifier = Modifier.height(20.dp))

            Text(
                text = "PocketCluster",
                fontSize = 30.sp,
                fontWeight = FontWeight.Bold,
                color = TextPrimary,
                letterSpacing = (-0.5).sp
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = "Decentralized personal storage",
                fontSize = 13.sp,
                color = TextMuted,
                letterSpacing = 0.2.sp
            )

            Spacer(modifier = Modifier.height(44.dp))

            // ── Registration card ─────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(22.dp))
                    .background(BgCard)
                    .border(
                        width = 1.dp,
                        brush = Brush.verticalGradient(
                            listOf(AccentCyan.copy(0.30f), BorderSubtle)
                        ),
                        shape = RoundedCornerShape(22.dp)
                    )
                    .padding(24.dp)
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {

                    Text(
                        text = "Connect Your Device",
                        fontSize = 17.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimary
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "Register this device to join the cluster and start storing or accessing files across the network.",
                        fontSize = 13.sp,
                        color = TextSecondary,
                        textAlign = TextAlign.Center,
                        lineHeight = 20.sp
                    )

                    Spacer(modifier = Modifier.height(22.dp))

                    // ── Status indicator ──────────────────────────────────────
                    val statusBg = when {
                        isError   -> ErrorRed.copy(alpha = 0.08f)
                        isSuccess -> AccentGreen.copy(alpha = 0.08f)
                        isLoading -> AccentCyan.copy(alpha = 0.07f)
                        else      -> BorderSubtle.copy(alpha = 0.5f)
                    }
                    val statusBorder = when {
                        isError   -> ErrorRed.copy(alpha = 0.35f)
                        isSuccess -> AccentGreen.copy(alpha = 0.35f)
                        isLoading -> AccentCyan.copy(alpha = 0.35f)
                        else      -> BorderSubtle
                    }
                    val statusDotColor = when {
                        isError   -> ErrorRed
                        isSuccess -> AccentGreen
                        isLoading -> AccentCyan
                        else      -> TextMuted
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(statusBg)
                            .border(1.dp, statusBorder, RoundedCornerShape(10.dp))
                            .padding(horizontal = 14.dp, vertical = 11.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        if (isLoading) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(12.dp),
                                color = AccentCyan,
                                strokeWidth = 1.5.dp
                            )
                        } else {
                            Box(
                                modifier = Modifier
                                    .size(7.dp)
                                    .clip(CircleShape)
                                    .background(statusDotColor)
                            )
                        }

                        Text(
                            text = statusText,
                            fontSize = 12.sp,
                            color = if (isError) ErrorRed else if (isSuccess) AccentGreen else TextSecondary,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    // ── Register button ───────────────────────────────────────
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(
                                if (!isLoading)
                                    Brush.horizontalGradient(
                                        listOf(AccentCyan.copy(0.22f), AccentPurple.copy(0.14f))
                                    )
                                else
                                    Brush.horizontalGradient(listOf(BgCard, BgCard))
                            )
                            .border(
                                1.dp,
                                if (!isLoading) AccentCyan.copy(0.55f) else BorderSubtle,
                                RoundedCornerShape(12.dp)
                            )
                    ) {
                        Button(
                            onClick = {
                                scope.launch {
                                    isLoading = true
                                    isError   = false
                                    isSuccess = false
                                    statusText = "connecting to cluster..."

                                    try {
                                        val request = DeviceRegistrationRequest(
                                            user_id           = 1,
                                            device_name       = DeviceInfoProvider.getDeviceName(),
                                            fingerprint       = DeviceInfoProvider.getDeviceFingerprint(context),
                                            storage_capacity  = DeviceInfoProvider.getTotalStorageBytes(),
                                            available_storage = DeviceInfoProvider.getAvailableStorageBytes()
                                        )

                                        val response = ApiClient.apiService.registerDevice(request)
                                        PreferencesManager.saveDeviceId(context, response.device_id)
                                        PreferencesManager.setRegistered(context, true)

                                        isSuccess  = true
                                        statusText = "registered (device_id = ${response.device_id})"
                                        onRegistered()

                                    } catch (e: Exception) {
                                        isError    = true
                                        statusText = "registration failed: ${e.message}"
                                        e.printStackTrace()
                                    } finally {
                                        isLoading = false
                                    }
                                }
                            },
                            enabled = !isLoading,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor         = Color.Transparent,
                                disabledContainerColor = Color.Transparent
                            ),
                            elevation = ButtonDefaults.buttonElevation(0.dp, 0.dp, 0.dp)
                        ) {
                            Text(
                                text = if (isLoading) "Registering..." else "Join Network",
                                fontSize = 15.sp,
                                fontWeight = FontWeight.SemiBold,
                                color = if (!isLoading) AccentCyan else TextMuted
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // ── Bottom trust line ─────────────────────────────────────────────
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(5.dp)
                        .clip(CircleShape)
                        .background(AccentGreen.copy(0.6f))
                )
                Text(
                    text = "End-to-end encrypted  ·  Distributed across trusted nodes",
                    fontSize = 11.sp,
                    color = TextMuted,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}