package com.phonecluster.app.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.phonecluster.app.ml.EmbeddingEngine
import android.util.Log
import androidx.compose.runtime.LaunchedEffect
import com.phonecluster.app.core.SERVER_BASE_URL
import com.phonecluster.app.ml.OnnxTokenizer
import com.phonecluster.app.ml.SummaryEngine
import com.phonecluster.app.storage.AppDatabase
import com.phonecluster.app.storage.FileEntity
import com.phonecluster.app.utils.ChunkUploader
import com.phonecluster.app.utils.ChunkedFileInfo
import com.phonecluster.app.utils.FileChunk
import com.phonecluster.app.utils.FileChunker
import com.phonecluster.app.utils.FileTextExtractor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.phonecluster.app.utils.ClusterStatusResponse
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun UserModeScreen(
    engine: EmbeddingEngine,
    onBackClick: () -> Unit = {},
    onSearchClick: () -> Unit,
    onBrowseClick: () -> Unit
) {
    val context = LocalContext.current
    val scope   = rememberCoroutineScope()

    val db  = AppDatabase.getDatabase(context)
    val dao = db.fileDao()

    // ── State ────────────────────────────────────────────────────────────────
    var selectedFileUri  by remember { mutableStateOf<Uri?>(null) }
    var fileInfo         by remember { mutableStateOf<ChunkedFileInfo?>(null) }
    var chunks           by remember { mutableStateOf<List<FileChunk>>(emptyList()) }
    var isChunking       by remember { mutableStateOf(false) }
    var chunkingProgress by remember { mutableStateOf(0 to 0) }
    var errorMessage     by remember { mutableStateOf<String?>(null) }
    var isUploading      by remember { mutableStateOf(false) }
    var uploadProgress   by remember { mutableStateOf(0 to 0) }
    var uploadedFileId   by remember { mutableStateOf<Int?>(null) }
    var clusterStatus    by remember { mutableStateOf<ClusterStatusResponse?>(null) }

    // ── Cluster status polling ────────────────────────────────────────────────
    LaunchedEffect(Unit) {
        val client = okhttp3.OkHttpClient()
        val gson   = com.google.gson.Gson()

        while (isActive) {
            try {
                val json = withContext(Dispatchers.IO) {
                    val request = okhttp3.Request.Builder()
                        .url("$SERVER_BASE_URL/cluster/status")
                        .get()
                        .build()
                    val response = client.newCall(request).execute()
                    response.body?.string()
                }
                if (json != null) {
                    clusterStatus = gson.fromJson(json, ClusterStatusResponse::class.java)
                }
            } catch (e: Exception) {
                Log.e("CLUSTER_STATUS", "Status fetch failed", e)
            }
            delay(5000)
        }
    }

    val totalClusterStorage = clusterStatus?.devices
        ?.filter { it.status == "ONLINE" }
        ?.sumOf { it.availableStorage } ?: 0L

    val usedClusterStorage = clusterStatus?.files
        ?.sumOf { it.fileSize } ?: 0L

    val onlineDevices = clusterStatus?.devices
        ?.count { it.status == "ONLINE" } ?: 0

    Log.d("STATUS_STORE", "${totalClusterStorage},${usedClusterStorage}")

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            selectedFileUri = it
            fileInfo        = FileChunker.getFileInfo(context, it)
            chunks          = emptyList()
            errorMessage    = null
            uploadedFileId  = null
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDeep)
    ) {
        LazyColumn(
            modifier       = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 32.dp)
        ) {

            // ── Top Bar ──────────────────────────────────────────────────────
            item {
                DashboardTopBar(onBackClick = onBackClick)
            }

            // ── Storage Arc + Devices ─────────────────────────────────────────
            item {
                Spacer(modifier = Modifier.height(4.dp))
                StorageStatsCard(
                    usedBytes  = usedClusterStorage,
                    totalBytes = totalClusterStorage,
                    nodeCount  = onlineDevices
                )
                Spacer(modifier = Modifier.height(20.dp))
            }

            // ── Quick Actions ─────────────────────────────────────────────────
            item {
                Text(
                    text          = "QUICK ACTIONS",
                    fontSize      = 10.sp,
                    color         = TextMuted,
                    letterSpacing = 1.2.sp,
                    fontFamily    = FontFamily.Monospace,
                    modifier      = Modifier.padding(horizontal = 16.dp)
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    QuickActionTile(
                        icon     = Icons.Outlined.CloudUpload,
                        label    = "Upload File",
                        sublabel = "Add new files",
                        color    = AccentCyan,
                        modifier = Modifier.weight(1f),
                        onClick  = { filePickerLauncher.launch("*/*") }
                    )
                    QuickActionTile(
                        icon     = Icons.Outlined.Search,
                        label    = "Search Files",
                        sublabel = "Find content",
                        color    = AccentPurple,
                        modifier = Modifier.weight(1f),
                        onClick  = onSearchClick
                    )
                    QuickActionTile(
                        icon     = Icons.Outlined.PhoneAndroid,
                        label    = "Browse Files",
                        sublabel = "Manage nodes",
                        color    = AccentGreen,
                        modifier = Modifier.weight(1f),
                        onClick  = onBrowseClick
                    )
                }
                Spacer(modifier = Modifier.height(20.dp))
            }

            // ── Status messages ───────────────────────────────────────────────
            errorMessage?.let { error ->
                item {
                    StatusBanner(
                        text     = error,
                        isError  = true,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }

            uploadedFileId?.let { id ->
                item {
                    StatusBanner(
                        text     = "File uploaded successfully · ID $id",
                        isError  = false,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }

            // ── Selected file card ────────────────────────────────────────────
            fileInfo?.let { info ->
                item {
                    Text(
                        text          = "SELECTED FILE",
                        fontSize      = 10.sp,
                        color         = TextMuted,
                        letterSpacing = 1.2.sp,
                        fontFamily    = FontFamily.Monospace,
                        modifier      = Modifier.padding(horizontal = 16.dp)
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    SelectedFileCard(info = info, modifier = Modifier.padding(horizontal = 16.dp))
                    Spacer(modifier = Modifier.height(12.dp))
                }

                // Chunk button
                item {
                    DarkActionButton(
                        label       = if (isChunking) "Chunking…" else "Prepare Chunks",
                        icon        = Icons.Outlined.Build,
                        accentColor = AccentAmber,
                        enabled     = !isChunking && chunks.isEmpty(),
                        modifier    = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp),
                        onClick = {
                            scope.launch {
                                isChunking   = true
                                errorMessage = null
                                try {
                                    chunks = withContext(Dispatchers.IO) {
                                        FileChunker.chunkFile(context, selectedFileUri!!) { cur, tot ->
                                            chunkingProgress = cur to tot
                                        }
                                    }
                                } catch (e: Exception) {
                                    errorMessage = "Chunking failed: ${e.message}"
                                    e.printStackTrace()
                                } finally {
                                    isChunking = false
                                }
                            }
                        }
                    )

                    if (isChunking) {
                        Spacer(modifier = Modifier.height(10.dp))
                        ProgressRow(
                            label    = "Processing chunk ${chunkingProgress.first}…",
                            progress = null,
                            modifier = Modifier.padding(horizontal = 16.dp)
                        )
                    }

                    Spacer(modifier = Modifier.height(10.dp))
                }

                // Upload button
                if (chunks.isNotEmpty()) {
                    item {
                        DarkActionButton(
                            label       = if (isUploading) "Uploading…" else "Upload to Cluster",
                            icon        = Icons.Outlined.CloudUpload,
                            accentColor = AccentCyan,
                            enabled     = !isUploading,
                            modifier    = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 16.dp),
                            onClick = {
                                scope.launch {
                                    isUploading    = true
                                    errorMessage   = null
                                    uploadedFileId = null
                                    uploadProgress = 0 to chunks.size
                                    try {
                                        val baseUrl = SERVER_BASE_URL
                                        val userId  = 1

                                        val fileId = withContext(Dispatchers.IO) {
                                            ChunkUploader.uploadAll(
                                                baseUrl  = baseUrl,
                                                userId   = userId,
                                                fileInfo = info,
                                                chunks   = chunks
                                            ) { uploaded, total ->
                                                uploadProgress = uploaded to total
                                            }
                                        }
                                        uploadedFileId = fileId
                                        withContext(Dispatchers.IO) {
                                            val fileText  = FileTextExtractor.extractText(context, selectedFileUri!!)
                                            Log.d("PDF_DEBUG", "Extracted text:\n$fileText")
                                            val summary   = SummaryEngine.summarize(fileText, 0.7f)
                                            val tokenizer = OnnxTokenizer(context)
                                            val (inputIds, attentionMask, tokenTypeIds) = tokenizer.tokenize(summary)
                                            val embedding = engine.generateEmbedding(inputIds, attentionMask, tokenTypeIds)
                                            Log.d("ROOM_TEST", "Inserting into Room DB")
                                            dao.insert(
                                                FileEntity(
                                                    serverFileId = fileId.toLong(),
                                                    fileName     = info.name,
                                                    fileType     = info.mimeType ?: "unknown",
                                                    fileDate     = System.currentTimeMillis(),
                                                    fileSize     = info.size,
                                                    embedding    = embedding
                                                )
                                            )
                                        }
                                    } catch (e: Exception) {
                                        errorMessage = "Upload failed: ${e.message}"
                                        e.printStackTrace()
                                    } finally {
                                        isUploading = false
                                    }
                                }
                            }
                        )

                        if (isUploading) {
                            Spacer(modifier = Modifier.height(10.dp))
                            ProgressRow(
                                label    = "Uploaded ${uploadProgress.first} / ${uploadProgress.second} chunks",
                                progress = uploadProgress.first.toFloat() / uploadProgress.second.toFloat(),
                                modifier = Modifier.padding(horizontal = 16.dp)
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                    }
                }
            }

            // ── Chunk list ────────────────────────────────────────────────────
            if (chunks.isNotEmpty()) {
                item {
                    Row(
                        modifier = Modifier.padding(horizontal = 16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text          = "CHUNKS",
                            fontSize      = 10.sp,
                            color         = TextMuted,
                            letterSpacing = 1.2.sp,
                            fontFamily    = FontFamily.Monospace
                        )
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(20.dp))
                                .background(AccentCyan.copy(alpha = 0.12f))
                                .padding(horizontal = 8.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text       = "${chunks.size}",
                                fontSize   = 11.sp,
                                color      = AccentCyan,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                items(chunks) { chunk ->
                    ChunkCard(
                        chunk    = chunk,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 3.dp)
                    )
                }
            }
        }
    }
}

// ─── Top Bar ──────────────────────────────────────────────────────────────────

@Composable
public fun DashboardTopBar(onBackClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .background(Brush.verticalGradient(listOf(Color(0xFF060E1E), BgDeep)))
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 8.dp, end = 16.dp, top = 12.dp, bottom = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = onBackClick) {
                Icon(
                    imageVector    = Icons.Default.ArrowBackIosNew,
                    contentDescription = "Back",
                    tint           = TextPrimary,
                    modifier       = Modifier.size(18.dp)
                )
            }

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text          = "PocketCluster",
                    fontSize      = 20.sp,
                    fontWeight    = FontWeight.Bold,
                    color         = TextPrimary,
                    letterSpacing = (-0.3).sp
                )
                Text(
                    text     = "Decentralized Personal Storage",
                    fontSize = 11.sp,
                    color    = TextMuted
                )
            }

            Row(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(AccentGreen.copy(alpha = 0.1f))
                    .border(1.dp, AccentGreen.copy(alpha = 0.3f), RoundedCornerShape(20.dp))
                    .padding(horizontal = 10.dp, vertical = 5.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(5.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(AccentGreen)
                )
                Text(
                    text       = "Cluster Online",
                    fontSize   = 11.sp,
                    fontWeight = FontWeight.Medium,
                    color      = AccentGreen
                )
            }
        }

        // Cyan hairline divider
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            Color.Transparent,
                            AccentCyan.copy(alpha = 0.35f),
                            AccentCyan.copy(alpha = 0.6f),
                            AccentCyan.copy(alpha = 0.35f),
                            Color.Transparent
                        )
                    )
                )
        )
    }
}

// ─── Storage Stats Card (Circular Arc + Node Count) ───────────────────────────

@Composable
private fun StorageStatsCard(usedBytes: Long, totalBytes: Long, nodeCount: Int) {
    val progress = if (totalBytes == 0L) 0f
    else (usedBytes.toFloat() / totalBytes.toFloat()).coerceIn(0f, 1f)

    val animatedProgress by animateFloatAsState(
        targetValue   = progress,
        animationSpec = tween(1200, easing = EaseOutCubic),
        label         = "arcProgress"
    )

    // Arc color transitions: green → amber → red by fill level
    val arcColor = when {
        progress < 0.6f -> AccentCyan
        progress < 0.85f -> AccentAmber
        else             -> ErrorRed
    }

    Card(
        modifier  = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        shape     = RoundedCornerShape(20.dp),
        colors    = CardDefaults.cardColors(containerColor = BgCard),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.horizontalGradient(
                        listOf(AccentCyan.copy(0.04f), Color.Transparent)
                    )
                )
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {

                // ── Circular Arc ──────────────────────────────────────────────
                Box(
                    modifier         = Modifier.size(110.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val sw        = 10.dp.toPx()
                        val arcSize   = Size(size.width - sw, size.height - sw)
                        val topLeft   = Offset(sw / 2f, sw / 2f)
                        val startAngle = 135f
                        val sweep      = 270f

                        // Track
                        drawArc(
                            color      = BorderSubtle,
                            startAngle = startAngle,
                            sweepAngle = sweep,
                            useCenter  = false,
                            topLeft    = topLeft,
                            size       = arcSize,
                            style      = Stroke(width = sw, cap = StrokeCap.Round)
                        )

                        // Progress arc
                        if (animatedProgress > 0f) {
                            drawArc(
                                brush      = Brush.sweepGradient(
                                    colors = listOf(
                                        arcColor.copy(alpha = 0.6f),
                                        arcColor
                                    ),
                                    center = center
                                ),
                                startAngle = startAngle,
                                sweepAngle = sweep * animatedProgress,
                                useCenter  = false,
                                topLeft    = topLeft,
                                size       = arcSize,
                                style      = Stroke(width = sw, cap = StrokeCap.Round)
                            )
                        }
                    }

                    // Center label
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text       = "%.0f%%".format(animatedProgress * 100),
                            fontSize   = 18.sp,
                            fontWeight = FontWeight.Bold,
                            color      = arcColor,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text          = "used",
                            fontSize      = 9.sp,
                            color         = TextMuted,
                            letterSpacing = 0.5.sp
                        )
                    }
                }

                Spacer(modifier = Modifier.width(16.dp))

                // ── Storage details + node count ──────────────────────────────
                Column(
                    modifier              = Modifier.weight(1f),
                    verticalArrangement   = Arrangement.spacedBy(12.dp)
                ) {
                    // Storage label
                    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            text          = "CLUSTER STORAGE",
                            fontSize      = 9.sp,
                            color         = TextMuted,
                            letterSpacing = 1.2.sp,
                            fontFamily    = FontFamily.Monospace
                        )
                        Row(
                            verticalAlignment = Alignment.Bottom,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text       = formatSize(usedBytes),
                                fontSize   = 18.sp,
                                fontWeight = FontWeight.Bold,
                                color      = AccentCyan,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                text       = "/ ${formatSize(totalBytes)}",
                                fontSize   = 12.sp,
                                color      = TextMuted,
                                fontFamily = FontFamily.Monospace,
                                modifier   = Modifier.padding(bottom = 1.dp)
                            )
                        }
                    }

                    // Divider
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(1.dp)
                            .background(BorderSubtle)
                    )

                    // Node count
                    Row(
                        verticalAlignment     = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(28.dp)
                                .clip(RoundedCornerShape(7.dp))
                                .background(AccentPurple.copy(0.1f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector    = Icons.Outlined.Hub,
                                contentDescription = null,
                                tint           = AccentPurple,
                                modifier       = Modifier.size(15.dp)
                            )
                        }

                        Column {
                            Text(
                                text       = "$nodeCount",
                                fontSize   = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color      = TextPrimary,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                text          = "active nodes",
                                fontSize      = 10.sp,
                                color         = TextMuted,
                                letterSpacing = 0.2.sp
                            )
                        }

                        Spacer(modifier = Modifier.weight(1f))

                        Row(
                            modifier = Modifier
                                .clip(RoundedCornerShape(20.dp))
                                .background(AccentGreen.copy(0.1f))
                                .border(1.dp, AccentGreen.copy(0.25f), RoundedCornerShape(20.dp))
                                .padding(horizontal = 8.dp, vertical = 3.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(5.dp)
                                    .clip(CircleShape)
                                    .background(AccentGreen)
                            )
                            Text(
                                text       = "ONLINE",
                                fontSize   = 9.sp,
                                fontWeight = FontWeight.Bold,
                                color      = AccentGreen,
                                fontFamily = FontFamily.Monospace,
                                letterSpacing = 0.5.sp
                            )
                        }
                    }
                }
            }
        }
    }
}

// ─── Quick Action Tiles ───────────────────────────────────────────────────────

@Composable
private fun QuickActionTile(
    icon: ImageVector,
    label: String,
    sublabel: String,
    color: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()

    val scale by animateFloatAsState(
        targetValue   = if (isPressed) 0.94f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessHigh),
        label         = "tileScale"
    )
    val bgAlpha by animateFloatAsState(
        targetValue   = if (isPressed) 0.18f else 0.08f,
        animationSpec = tween(100),
        label         = "tileBg"
    )

    Card(
        modifier = modifier
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .clickable(interactionSource = interactionSource, indication = null) { onClick() },
        shape     = RoundedCornerShape(14.dp),
        colors    = CardDefaults.cardColors(containerColor = BgCard),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = androidx.compose.foundation.BorderStroke(
            1.dp, color.copy(alpha = if (isPressed) 0.4f else 0.15f)
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(listOf(color.copy(bgAlpha), Color.Transparent))
                )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 16.dp, horizontal = 8.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(44.dp)
                        .clip(RoundedCornerShape(11.dp))
                        .background(color.copy(0.12f))
                        .border(1.dp, color.copy(0.2f), RoundedCornerShape(11.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector    = icon,
                        contentDescription = label,
                        tint           = color,
                        modifier       = Modifier.size(22.dp)
                    )
                }

                Text(
                    text      = label,
                    fontSize  = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    color     = TextPrimary,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                    lineHeight = 15.sp
                )

                Text(
                    text      = sublabel,
                    fontSize  = 10.sp,
                    color     = TextMuted,
                    textAlign = androidx.compose.ui.text.style.TextAlign.Center
                )
            }
        }
    }
}

// ─── Selected File Card ───────────────────────────────────────────────────────

@Composable
private fun SelectedFileCard(info: ChunkedFileInfo, modifier: Modifier = Modifier) {
    Card(
        modifier  = modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(14.dp),
        colors    = CardDefaults.cardColors(containerColor = BgCard),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = androidx.compose.foundation.BorderStroke(
            1.dp,
            Brush.horizontalGradient(listOf(AccentCyan.copy(0.4f), BorderSubtle, BorderSubtle))
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.horizontalGradient(listOf(AccentCyan.copy(0.05f), Color.Transparent))
                )
                .padding(14.dp)
        ) {
            Row(
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(AccentCyan.copy(0.1f)),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector    = Icons.Outlined.InsertDriveFile,
                        contentDescription = null,
                        tint           = AccentCyan,
                        modifier       = Modifier.size(22.dp)
                    )
                }

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text       = info.name,
                        fontSize   = 14.sp,
                        fontWeight = FontWeight.SemiBold,
                        color      = TextPrimary,
                        maxLines   = 1,
                        overflow   = TextOverflow.Ellipsis
                    )
                    Spacer(modifier = Modifier.height(3.dp))
                    Text(
                        text       = "${formatSize(info.size)}  ·  ${info.mimeType ?: "Unknown type"}",
                        fontSize   = 11.sp,
                        color      = TextSecondary,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))
            HorizontalDivider(color = BorderSubtle, thickness = 1.dp)
            Spacer(modifier = Modifier.height(10.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                MetaItem(label = "Chunks", value = "${info.totalChunks}")
                MetaItem(label = "Chunk size", value = "40 KB")
                MetaItem(label = "Total size", value = formatSize(info.size))
            }
        }
    }
}

@Composable
private fun MetaItem(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text       = value,
            fontSize   = 15.sp,
            fontWeight = FontWeight.Bold,
            color      = TextPrimary,
            fontFamily = FontFamily.Monospace
        )
        Text(text = label, fontSize = 10.sp, color = TextMuted)
    }
}

// ─── Dark Action Button ───────────────────────────────────────────────────────

@Composable
private fun DarkActionButton(
    label: String,
    icon: ImageVector,
    accentColor: Color,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()

    val scale by animateFloatAsState(
        targetValue   = if (isPressed && enabled) 0.97f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessHigh),
        label         = "btnScale"
    )

    Box(
        modifier = modifier
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .clip(RoundedCornerShape(12.dp))
            .background(
                if (enabled)
                    Brush.horizontalGradient(listOf(accentColor.copy(0.15f), accentColor.copy(0.07f)))
                else
                    Brush.horizontalGradient(listOf(BgCard, BgCard))
            )
            .border(
                1.dp,
                if (enabled) accentColor.copy(if (isPressed) 0.7f else 0.35f) else BorderSubtle,
                RoundedCornerShape(12.dp)
            )
            .clickable(
                interactionSource = interactionSource,
                indication        = null,
                enabled           = enabled
            ) { onClick() }
            .padding(vertical = 14.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Icon(
                imageVector    = icon,
                contentDescription = null,
                tint           = if (enabled) accentColor else TextMuted,
                modifier       = Modifier.size(18.dp)
            )
            Text(
                text       = label,
                fontSize   = 14.sp,
                fontWeight = FontWeight.SemiBold,
                color      = if (enabled) TextPrimary else TextMuted
            )
        }
    }
}

// ─── Progress Row ─────────────────────────────────────────────────────────────

@Composable
private fun ProgressRow(label: String, progress: Float?, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxWidth()) {
        Text(text = label, fontSize = 11.sp, color = TextSecondary)
        Spacer(modifier = Modifier.height(6.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(BorderSubtle)
        ) {
            if (progress != null) {
                val animProg by animateFloatAsState(
                    targetValue   = progress,
                    animationSpec = tween(300),
                    label         = "prog"
                )
                Box(
                    modifier = Modifier
                        .fillMaxWidth(animProg)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(2.dp))
                        .background(
                            Brush.horizontalGradient(listOf(AccentCyan, AccentCyan.copy(0.6f)))
                        )
                )
            } else {
                // Indeterminate shimmer
                val inf = rememberInfiniteTransition(label = "shimmer")
                val x by inf.animateFloat(
                    initialValue  = -1f,
                    targetValue   = 2f,
                    animationSpec = infiniteRepeatable(tween(1000, easing = LinearEasing)),
                    label         = "shimX"
                )
                Box(
                    modifier = Modifier
                        .fillMaxWidth(0.35f)
                        .fillMaxHeight()
                        .graphicsLayer { translationX = x * 300 }
                        .clip(RoundedCornerShape(2.dp))
                        .background(
                            Brush.horizontalGradient(
                                listOf(Color.Transparent, AccentCyan, Color.Transparent)
                            )
                        )
                )
            }
        }
    }
}

// ─── Status Banner ────────────────────────────────────────────────────────────

@Composable
private fun StatusBanner(text: String, isError: Boolean, modifier: Modifier = Modifier) {
    val color = if (isError) ErrorRed else AccentGreen
    val icon  = if (isError) Icons.Outlined.ErrorOutline else Icons.Outlined.CheckCircle

    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(color.copy(0.08f))
            .border(1.dp, color.copy(0.25f), RoundedCornerShape(10.dp))
            .padding(12.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Icon(imageVector = icon, contentDescription = null, tint = color, modifier = Modifier.size(18.dp))
        Text(text = text, fontSize = 13.sp, color = color, modifier = Modifier.weight(1f))
    }
}

// ─── Chunk Card ───────────────────────────────────────────────────────────────

@Composable
private fun ChunkCard(chunk: FileChunk, modifier: Modifier = Modifier) {
    Card(
        modifier  = modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(10.dp),
        colors    = CardDefaults.cardColors(containerColor = BgCard),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(28.dp)
                        .clip(RoundedCornerShape(6.dp))
                        .background(AccentCyan.copy(0.08f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text       = "${chunk.index}",
                        fontSize   = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color      = AccentCyan,
                        fontFamily = FontFamily.Monospace
                    )
                }

                Column {
                    Text(
                        text       = "Chunk #${chunk.index}",
                        fontSize   = 13.sp,
                        fontWeight = FontWeight.Medium,
                        color      = TextPrimary
                    )
                    Text(
                        text       = FileChunker.formatFileSize(chunk.size.toLong()),
                        fontSize   = 11.sp,
                        color      = TextMuted,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(20.dp))
                    .background(AccentGreen.copy(0.1f))
                    .border(1.dp, AccentGreen.copy(0.25f), RoundedCornerShape(20.dp))
                    .padding(horizontal = 10.dp, vertical = 3.dp)
            ) {
                Text(
                    text       = "Ready",
                    fontSize   = 10.sp,
                    fontWeight = FontWeight.Medium,
                    color      = AccentGreen
                )
            }
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

private fun formatSize(bytes: Long): String = when {
    bytes >= 1_073_741_824L -> "%.1f GB".format(bytes / 1_073_741_824.0)
    bytes >= 1_048_576L     -> "%.1f MB".format(bytes / 1_048_576.0)
    bytes >= 1_024L         -> "%.1f KB".format(bytes / 1_024.0)
    else                    -> "$bytes B"
}