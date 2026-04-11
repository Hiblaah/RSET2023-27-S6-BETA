package com.phonecluster.app.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.phonecluster.app.storage.FileEntity
import java.text.SimpleDateFormat
import java.util.*
import com.phonecluster.app.screens.FileBrowserViewModel

// ─── Color Tokens ────────────────────────────────────────────────────────────

private val BgDeep        = Color(0xFF020617)
private val BgCard        = Color(0xFF0D1424)
private val BgCardHover   = Color(0xFF111827)
private val BgIconPdf     = Color(0xFF1A1030)
private val BgIconImg     = Color(0xFF0A1F1A)
private val BgIconDoc     = Color(0xFF0A1525)
private val BgIconGeneric = Color(0xFF111827)

private val AccentCyan    = Color(0xFF22D3EE)
private val AccentPurple  = Color(0xFFA78BFA)
private val AccentGreen   = Color(0xFF34D399)
private val AccentAmber   = Color(0xFFFBBF24)

private val TextPrimary   = Color(0xFFF1F5F9)
private val TextSecondary = Color(0xFF94A3B8)
private val TextMuted     = Color(0xFF475569)

private val BorderSubtle  = Color(0xFF1E293B)
private val ErrorRed      = Color(0xFFEF4444)

// ─── Helpers ─────────────────────────────────────────────────────────────────

private fun formatDate(epochMs: Long): String {
    val sdf = SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
    return sdf.format(Date(epochMs))
}

private fun formatSize(bytes: Long): String = when {
    bytes >= 1_073_741_824L -> "%.1f GB".format(bytes / 1_073_741_824.0)
    bytes >= 1_048_576L     -> "%.1f MB".format(bytes / 1_048_576.0)
    bytes >= 1_024L         -> "%.1f KB".format(bytes / 1_024.0)
    else                    -> "$bytes B"
}

private data class FileTypeStyle(
    val icon: ImageVector,
    val tint: Color,
    val bgColor: Color,
    val label: String
)

private fun resolveFileType(fileType: String, fileName: String): FileTypeStyle {
    val ext = fileName.substringAfterLast('.', fileType).lowercase()
    return when (ext) {
        "pdf"                          -> FileTypeStyle(Icons.Default.Description,    Color(0xFFFC8181), BgIconPdf,        "PDF")
        "jpg", "jpeg", "png",
        "gif", "webp", "bmp", "heic"  -> FileTypeStyle(Icons.Default.Image,          AccentGreen,       BgIconImg,        "Image")
        "doc", "docx", "txt", "md"    -> FileTypeStyle(Icons.Default.Article,         AccentCyan,        BgIconDoc,        "Document")
        "mp4", "mov", "avi", "mkv"    -> FileTypeStyle(Icons.Default.PlayCircle,      AccentPurple,      Color(0xFF150D2A), "Video")
        "mp3", "wav", "flac", "aac"   -> FileTypeStyle(Icons.Default.MusicNote,       AccentAmber,       Color(0xFF1A1200), "Audio")
        "zip", "tar", "gz", "rar"     -> FileTypeStyle(Icons.Default.FolderZip,       AccentAmber,       Color(0xFF1A1000), "Archive")
        "xls", "xlsx", "csv"          -> FileTypeStyle(Icons.Default.TableChart,      AccentGreen,       Color(0xFF051A10), "Spreadsheet")
        else                          -> FileTypeStyle(Icons.Default.InsertDriveFile, TextSecondary,     BgIconGeneric,    "File")
    }
}

// ─── Screen ──────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FileBrowserScreen(
    onBackClick: () -> Unit,
    viewModel: FileBrowserViewModel = viewModel()
) {
    val files by viewModel.files.collectAsState()

    Scaffold(
        containerColor = BgDeep,
        topBar = {
            FileBrowserTopBar(onBackClick = onBackClick, fileCount = files.size)
        }
    ) { paddingValues ->

        if (files.isEmpty()) {
            EmptyState(modifier = Modifier.padding(paddingValues))
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                contentPadding = PaddingValues(
                    start  = 16.dp,
                    end    = 16.dp,
                    top    = 12.dp,
                    bottom = 32.dp
                ),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                item {
                    StorageSummaryBanner(files = files)
                    Spacer(modifier = Modifier.height(8.dp))
                }

                itemsIndexed(files) { _, file ->
                    FileItem(
                        file       = file,
                        onDownload = { id -> viewModel.downloadFile(id) },
                        onDelete   = { id -> viewModel.deleteFile(id) }
                    )
                }
            }
        }
    }
}

// ─── Top App Bar ─────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun FileBrowserTopBar(
    onBackClick: () -> Unit,
    fileCount: Int
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Brush.verticalGradient(listOf(Color(0xFF060E1E), BgDeep)))
    ) {
        TopAppBar(
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor             = Color.Transparent,
                navigationIconContentColor = TextPrimary,
                actionIconContentColor     = TextSecondary,
                titleContentColor          = TextPrimary
            ),
            navigationIcon = {
                IconButton(onClick = onBackClick) {
                    Icon(
                        imageVector    = Icons.Default.ArrowBackIosNew,
                        contentDescription = "Back",
                        modifier       = Modifier.size(20.dp)
                    )
                }
            },
            title = {
                Column {
                    Text(
                        text          = "My Files",
                        fontSize      = 20.sp,
                        fontWeight    = FontWeight.Bold,
                        color         = TextPrimary,
                        letterSpacing = (-0.3).sp
                    )
                    Text(
                        text     = "$fileCount items  ·  PocketCluster",
                        fontSize = 11.sp,
                        color    = TextMuted
                    )
                }
            }
        )

        // Cyan accent divider
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

// ─── Storage Summary Banner ───────────────────────────────────────────────────

@Composable
private fun StorageSummaryBanner(files: List<FileEntity>) {
    val totalBytes  = files.sumOf { it.fileSize }
    val uniqueTypes = files.map { it.fileName.substringAfterLast('.').lowercase() }.distinct().size

    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(16.dp),
        colors    = CardDefaults.cardColors(containerColor = Color(0xFF0A1628)),
        elevation = CardDefaults.cardElevation(0.dp),
        border    = androidx.compose.foundation.BorderStroke(
            1.dp,
            Brush.horizontalGradient(
                listOf(AccentCyan.copy(0.25f), BorderSubtle, AccentPurple.copy(0.15f))
            )
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            AccentCyan.copy(alpha = 0.06f),
                            Color.Transparent,
                            AccentPurple.copy(alpha = 0.04f)
                        )
                    )
                )
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 18.dp, vertical = 16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Total size
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text          = "TOTAL STORED",
                        fontSize      = 9.sp,
                        color         = TextMuted,
                        letterSpacing = 1.2.sp,
                        fontFamily    = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text          = formatSize(totalBytes),
                        fontSize      = 26.sp,
                        fontWeight    = FontWeight.Bold,
                        color         = AccentCyan,
                        fontFamily    = FontFamily.Monospace,
                        letterSpacing = (-0.5).sp
                    )
                }

                // Divider
                Box(
                    modifier = Modifier
                        .width(1.dp)
                        .height(40.dp)
                        .background(BorderSubtle)
                )

                Spacer(modifier = Modifier.width(20.dp))

                // Files count
                SummaryChip(
                    label    = "${files.size}",
                    sublabel = "files",
                    color    = AccentCyan
                )

                Spacer(modifier = Modifier.width(20.dp))

                // Types count
                SummaryChip(
                    label    = "$uniqueTypes",
                    sublabel = "types",
                    color    = AccentPurple
                )
            }
        }
    }
}

@Composable
private fun SummaryChip(label: String, sublabel: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text          = label,
            fontSize      = 20.sp,
            fontWeight    = FontWeight.Bold,
            color         = color,
            fontFamily    = FontFamily.Monospace
        )
        Text(
            text          = sublabel,
            fontSize      = 10.sp,
            color         = TextMuted,
            letterSpacing = 0.5.sp
        )
    }
}

// ─── File Item ────────────────────────────────────────────────────────────────

@Composable
fun FileItem(
    file: FileEntity,
    onDownload: (Long) -> Unit,
    onDelete: (Long) -> Unit
) {
    val typeStyle = resolveFileType(file.fileType, file.fileName)
    var pressed   by remember { mutableStateOf(false) }
    val cardColor by animateColorAsState(
        targetValue  = if (pressed) BgCardHover else BgCard,
        animationSpec = tween(150),
        label        = "cardColor"
    )

    Card(
        modifier  = Modifier.fillMaxWidth(),
        shape     = RoundedCornerShape(14.dp),
        colors    = CardDefaults.cardColors(containerColor = cardColor),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        border    = androidx.compose.foundation.BorderStroke(
            width = 1.dp,
            brush = Brush.horizontalGradient(
                listOf(
                    typeStyle.tint.copy(alpha = 0.15f),
                    BorderSubtle
                )
            )
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {

            // ── File type icon ────────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .size(46.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(typeStyle.bgColor)
                    .border(1.dp, typeStyle.tint.copy(0.15f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector    = typeStyle.icon,
                    contentDescription = typeStyle.label,
                    tint           = typeStyle.tint,
                    modifier       = Modifier.size(24.dp)
                )
            }

            Spacer(modifier = Modifier.width(14.dp))

            // ── File name + metadata ──────────────────────────────────────────
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text          = file.fileName,
                    fontSize      = 14.sp,
                    fontWeight    = FontWeight.SemiBold,
                    color         = TextPrimary,
                    maxLines      = 1,
                    overflow      = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
                    letterSpacing = (-0.1).sp
                )

                Spacer(modifier = Modifier.height(4.dp))

                Row(
                    verticalAlignment      = Alignment.CenterVertically,
                    horizontalArrangement  = Arrangement.spacedBy(6.dp)
                ) {
                    // Type badge
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(typeStyle.tint.copy(0.1f))
                            .padding(horizontal = 5.dp, vertical = 1.dp)
                    ) {
                        Text(
                            text          = typeStyle.label.uppercase(),
                            fontSize      = 9.sp,
                            color         = typeStyle.tint,
                            fontFamily    = FontFamily.Monospace,
                            letterSpacing = 0.5.sp,
                            fontWeight    = FontWeight.Bold
                        )
                    }

                    Box(
                        modifier = Modifier
                            .size(3.dp)
                            .clip(CircleShape)
                            .background(TextMuted)
                    )

                    MetaChip(text = formatSize(file.fileSize), color = TextSecondary)

                    Box(
                        modifier = Modifier
                            .size(3.dp)
                            .clip(CircleShape)
                            .background(TextMuted)
                    )

                    Text(
                        text     = formatDate(file.fileDate),
                        fontSize = 11.sp,
                        color    = TextMuted
                    )
                }
            }

            Spacer(modifier = Modifier.width(4.dp))

            // ── Actions ───────────────────────────────────────────────────────
            Row(horizontalArrangement = Arrangement.spacedBy(0.dp)) {

                IconButton(
                    onClick  = { onDownload(file.serverFileId.toLong()) },
                    modifier = Modifier.size(38.dp)
                ) {
                    Icon(
                        imageVector    = Icons.Default.FileDownload,
                        contentDescription = "Download",
                        tint           = AccentCyan,
                        modifier       = Modifier.size(20.dp)
                    )
                }

                IconButton(
                    onClick = { onDelete(file.serverFileId.toLong()) },
                    modifier = Modifier.size(38.dp)
                ) {
                    Icon(
                        imageVector    = Icons.Default.DeleteOutline,
                        contentDescription = "Delete",
                        tint           = ErrorRed.copy(alpha = 0.75f),
                        modifier       = Modifier.size(20.dp)
                    )
                }
            }
        }
    }
}

// ─── Small UI Atoms ──────────────────────────────────────────────────────────

@Composable
private fun MetaChip(text: String, color: Color) {
    Text(
        text       = text,
        fontSize   = 11.sp,
        color      = color,
        fontFamily = FontFamily.Monospace
    )
}

// ─── Empty State ──────────────────────────────────────────────────────────────

@Composable
private fun EmptyState(modifier: Modifier = Modifier) {
    Box(
        modifier         = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .clip(RoundedCornerShape(22.dp))
                    .background(BgCard)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(22.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector    = Icons.Outlined.FolderOpen,
                    contentDescription = null,
                    tint           = TextMuted,
                    modifier       = Modifier.size(38.dp)
                )
            }

            Text(
                text       = "No files stored",
                fontSize   = 16.sp,
                fontWeight = FontWeight.SemiBold,
                color      = TextSecondary
            )

            Text(
                text      = "Files synced to your cluster\nwill appear here",
                fontSize  = 13.sp,
                color     = TextMuted,
                textAlign = TextAlign.Center
            )
        }
    }
}