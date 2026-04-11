package com.phonecluster.app.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.FileDownload
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.outlined.Hub
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.BorderStroke
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import com.phonecluster.app.ml.EmbeddingEngine
import com.phonecluster.app.ml.OnnxTokenizer
import com.phonecluster.app.ml.SimilarityUtils
import com.phonecluster.app.storage.AppDatabase
import androidx.lifecycle.viewmodel.compose.viewModel
// ─── Color Tokens ─────────────────────────────────────────────────────────────

private val BgDeep      = Color(0xFF020617)
private val BgCard      = Color(0xFF0D1424)
private val BorderSubtle= Color(0xFF1E293B)
private val AccentCyan  = Color(0xFF22D3EE)
private val AccentPurple= Color(0xFFA78BFA)
private val TextPrimary = Color(0xFFF1F5F9)
private val TextMuted   = Color(0xFF475569)
private val TextSecondary = Color(0xFF94A3B8)

// ─── Data Models ──────────────────────────────────────────────────────────────

data class ChatMessage(
    val text: String? = null,
    val results: List<SearchResult>? = null,
    val isUserQuery: Boolean = false
)

data class SearchResult(
    val serverfileId: Long,
    val fileName: String,
    val fileType: String,
    val score: Float
)

// ─── Screen ───────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SearchScreen(
    engine: EmbeddingEngine,
    onBackClick: () -> Unit
) {
    val context    = LocalContext.current
    val scope      = rememberCoroutineScope()
    val listState  = rememberLazyListState()

    var query      by remember { mutableStateOf("") }
    var messages   by remember { mutableStateOf<List<ChatMessage>>(emptyList()) }
    var isSearching by remember { mutableStateOf(false) }

    val db  = AppDatabase.getDatabase(context)
    val dao = db.fileDao()

    val viewModel: FileBrowserViewModel = viewModel()

    // Auto-scroll to bottom on new messages
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    Scaffold(
        containerColor = BgDeep,
        topBar = {
            SearchTopBar(onBackClick = onBackClick)
        }
    ) { padding ->

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {

            // ── Message feed ──────────────────────────────────────────────────
            if (messages.isEmpty()) {
                SearchEmptyState(modifier = Modifier.weight(1f))
            } else {
                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 16.dp),
                    state = listState,
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    contentPadding = PaddingValues(vertical = 16.dp)
                ) {
                    items(messages) { message ->
                        when {
                            message.isUserQuery && message.text != null ->
                                UserQueryBubble(text = message.text)

                            !message.isUserQuery && message.text != null && message.results == null ->
                                SystemLabel(text = message.text)

                            message.results != null -> {
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    message.results.forEach { result ->
                                        FileResultCard(
                                            result = result,
                                            onDownload = { id ->
                                                viewModel.downloadFile(id)
                                            }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ── Input bar ─────────────────────────────────────────────────────
            SearchInputBar(
                query       = query,
                isSearching = isSearching,
                onQueryChange = { query = it },
                onSend = {
                    if (query.isBlank()) return@SearchInputBar
                    val currentQuery = query
                    query = ""

                    scope.launch {
                        isSearching = true

                        // Append user message immediately
                        messages = messages + ChatMessage(
                            text        = currentQuery,
                            isUserQuery = true
                        )

                        try {
                            // ── STEP 1: Tokenization ──────────────────────
                            val t0 = System.currentTimeMillis()
                            val tokenizer = OnnxTokenizer(context)
                            val (inputIds, attentionMask, tokenTypeIds) =
                                tokenizer.tokenize(currentQuery)
                            val t1 = System.currentTimeMillis()
                            android.util.Log.d("PIPELINE_TIMING", "Tokenization: ${t1 - t0} ms")

                            // ── STEP 2: ONNX Embedding ────────────────────
                            val queryEmbedding = engine.generateEmbedding(
                                inputIds, attentionMask, tokenTypeIds
                            )
                            val t2 = System.currentTimeMillis()
                            android.util.Log.d("PIPELINE_TIMING", "ONNX Embedding inference: ${t2 - t1} ms")

                            // ── STEP 3: Room DB Fetch ─────────────────────
                            val files = withContext(Dispatchers.IO) {
                                dao.getAllFilesOnce()
                            }
                            val t3 = System.currentTimeMillis()
                            android.util.Log.d("PIPELINE_TIMING", "Room DB fetch (${files.size} files): ${t3 - t2} ms")

                            // ── STEP 4: Cosine Similarity Ranking ─────────
                            val ranked = files
                                .map { it to SimilarityUtils.cosineSimilarity(queryEmbedding, it.embedding) }
                                .sortedByDescending { it.second }
                                .take(3)
                            val t4 = System.currentTimeMillis()
                            android.util.Log.d("PIPELINE_TIMING", "Cosine similarity + ranking: ${t4 - t3} ms")
                            android.util.Log.d("PIPELINE_TIMING", "TOTAL search pipeline: ${t4 - t0} ms")

                            val resultList = ranked.map { (file, score) ->
                                SearchResult(
                                    serverfileId   = file.serverFileId,
                                    fileName = file.fileName,
                                    fileType = file.fileType,
                                    score    = score
                                )
                            }

                            messages = messages + ChatMessage(
                                text        = "Top matches",
                                results     = resultList,
                                isUserQuery = false
                            )

                        } catch (e: Exception) {
                            messages = messages + ChatMessage(
                                text        = "Search failed: ${e.message}",
                                isUserQuery = false
                            )
                        } finally {
                            isSearching = false
                        }
                    }
                }
            )
        }
    }
}

// ─── Top Bar ──────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SearchTopBar(onBackClick: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(listOf(Color(0xFF060E1E), BgDeep))
            )
    ) {
        TopAppBar(
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor          = Color.Transparent,
                navigationIconContentColor = TextPrimary,
                titleContentColor       = TextPrimary
            ),
            navigationIcon = {
                IconButton(onClick = onBackClick) {
                    Icon(
                        Icons.Default.ArrowBack,
                        contentDescription = "Back",
                        modifier = Modifier.size(20.dp)
                    )
                }
            },
            title = {
                Column {
                    Text(
                        text       = "Smart Search",
                        fontSize   = 18.sp,
                        fontWeight = FontWeight.Bold,
                        color      = TextPrimary,
                        letterSpacing = (-0.3).sp
                    )
                    Text(
                        text     = "AI-powered file discovery",
                        fontSize = 11.sp,
                        color    = TextMuted
                    )
                }
            }
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            Color.Transparent,
                            AccentPurple.copy(0.35f),
                            AccentPurple.copy(0.6f),
                            AccentPurple.copy(0.35f),
                            Color.Transparent
                        )
                    )
                )
        )
    }
}

// ─── Empty State ──────────────────────────────────────────────────────────────

@Composable
private fun SearchEmptyState(modifier: Modifier = Modifier) {
    Box(
        modifier          = modifier.fillMaxWidth(),
        contentAlignment  = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(72.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(
                        Brush.linearGradient(
                            listOf(AccentPurple.copy(0.15f), AccentCyan.copy(0.08f))
                        )
                    )
                    .border(1.dp, AccentPurple.copy(0.3f), RoundedCornerShape(20.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector    = Icons.Outlined.Hub,
                    contentDescription = null,
                    tint           = AccentPurple,
                    modifier       = Modifier.size(34.dp)
                )
            }

            Text(
                text       = "Ask anything about your files",
                fontSize   = 15.sp,
                fontWeight = FontWeight.SemiBold,
                color      = TextSecondary
            )

            Text(
                text      = "\"Find my invoice PDFs\"\n\"Documents about the Q3 project\"",
                fontSize  = 12.sp,
                color     = TextMuted,
                textAlign = TextAlign.Center,
                lineHeight = 18.sp
            )
        }
    }
}

// ─── User Query Bubble ────────────────────────────────────────────────────────

@Composable
private fun UserQueryBubble(text: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.End
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 280.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 16.dp, topEnd = 4.dp,
                        bottomStart = 16.dp, bottomEnd = 16.dp
                    )
                )
                .background(
                    Brush.horizontalGradient(
                        listOf(AccentPurple.copy(0.25f), AccentCyan.copy(0.18f))
                    )
                )
                .border(
                    1.dp,
                    Brush.horizontalGradient(
                        listOf(AccentPurple.copy(0.5f), AccentCyan.copy(0.35f))
                    ),
                    RoundedCornerShape(
                        topStart = 16.dp, topEnd = 4.dp,
                        bottomStart = 16.dp, bottomEnd = 16.dp
                    )
                )
                .padding(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                    imageVector    = Icons.Default.Search,
                    contentDescription = null,
                    tint           = AccentPurple,
                    modifier       = Modifier.size(15.dp)
                )
                Text(
                    text       = text,
                    fontSize   = 14.sp,
                    color      = TextPrimary
                )
            }
        }
    }
}

// ─── System Label (e.g. "Top matches") ───────────────────────────────────────

@Composable
private fun SystemLabel(text: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Box(
            modifier = Modifier
                .size(4.dp)
                .clip(CircleShape)
                .background(TextMuted)
        )
        Text(
            text          = text,
            fontSize      = 11.sp,
            color         = TextMuted,
            letterSpacing = 0.5.sp
        )
        Box(
            modifier = Modifier
                .weight(1f)
                .height(1.dp)
                .background(BorderSubtle)
        )
    }
}

// ─── File Result Card ─────────────────────────────────────────────────────────

@Composable
fun FileResultCard(
    result: SearchResult,
    onDownload: (Long) -> Unit
) {

    val scoreColor = when {
        result.score >= 0.8f -> Color(0xFF34D399)
        result.score >= 0.5f -> AccentCyan
        else                 -> Color(0xFF94A3B8)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape    = RoundedCornerShape(14.dp),
        colors   = CardDefaults.cardColors(containerColor = BgCard),
        border   = BorderStroke(1.dp, BorderSubtle)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    Brush.horizontalGradient(
                        listOf(AccentCyan.copy(0.04f), Color.Transparent)
                    )
                )
                .padding(horizontal = 14.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // File icon
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xFF1A1030))
                    .border(1.dp, AccentPurple.copy(0.2f), RoundedCornerShape(10.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector    = Icons.Default.Description,
                    contentDescription = null,
                    tint           = Color(0xFFFC8181),
                    modifier       = Modifier.size(22.dp)
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text       = result.fileName,
                    fontSize   = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color      = TextPrimary,
                    maxLines   = 1,
                    overflow   = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(4.dp))

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text         = result.fileType.uppercase(),
                        fontSize     = 10.sp,
                        color        = TextMuted,
                        fontFamily   = FontFamily.Monospace,
                        letterSpacing = 0.5.sp
                    )

                    Box(
                        modifier = Modifier
                            .size(3.dp)
                            .clip(CircleShape)
                            .background(TextMuted)
                    )

                    // Score badge
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(scoreColor.copy(0.12f))
                            .border(1.dp, scoreColor.copy(0.3f), RoundedCornerShape(4.dp))
                            .padding(horizontal = 5.dp, vertical = 1.dp)
                    ) {
                        Text(
                            text       = "%.0f%%".format(result.score * 100),
                            fontSize   = 10.sp,
                            color      = scoreColor,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            IconButton(
                onClick  = { onDownload(result.serverfileId) },
                modifier = Modifier.size(36.dp)
            ) {
                Icon(
                    imageVector    = Icons.Default.FileDownload,
                    contentDescription = "Download",
                    tint           = AccentCyan,
                    modifier       = Modifier.size(20.dp)
                )
            }
        }
    }
}

// ─── Search Input Bar ─────────────────────────────────────────────────────────

@Composable
private fun SearchInputBar(
    query: String,
    isSearching: Boolean,
    onQueryChange: (String) -> Unit,
    onSend: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(listOf(Color.Transparent, BgDeep.copy(0.95f)))
            )
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(Color(0xFF0D1424))
                .border(
                    1.dp,
                    Brush.horizontalGradient(
                        listOf(AccentPurple.copy(0.35f), AccentCyan.copy(0.25f))
                    ),
                    RoundedCornerShape(14.dp)
                )
                .padding(horizontal = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value         = query,
                onValueChange = onQueryChange,
                modifier      = Modifier.weight(1f),
                placeholder   = {
                    Text(
                        "Ask about your files...",
                        color  = TextMuted,
                        fontSize = 14.sp
                    )
                },
                colors = TextFieldDefaults.colors(
                    focusedContainerColor   = Color.Transparent,
                    unfocusedContainerColor = Color.Transparent,
                    disabledContainerColor  = Color.Transparent,
                    focusedIndicatorColor   = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedTextColor        = TextPrimary,
                    unfocusedTextColor      = TextPrimary
                )
            )

            if (isSearching) {
                CircularProgressIndicator(
                    modifier    = Modifier
                        .size(36.dp)
                        .padding(8.dp),
                    color       = AccentCyan,
                    strokeWidth = 2.dp
                )
            } else {
                IconButton(
                    onClick  = onSend,
                    modifier = Modifier.size(40.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(32.dp)
                            .clip(RoundedCornerShape(8.dp))
                            .background(
                                Brush.linearGradient(
                                    listOf(AccentCyan.copy(0.25f), AccentPurple.copy(0.15f))
                                )
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector    = Icons.Default.Send,
                            contentDescription = "Search",
                            tint           = AccentCyan,
                            modifier       = Modifier.size(16.dp)
                        )
                    }
                }
            }
        }
    }
}