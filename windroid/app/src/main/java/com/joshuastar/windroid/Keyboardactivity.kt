package com.joshuastar.windroid

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.KeyEvent
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.activity.ComponentActivity

class KeyboardActivity : ComponentActivity() {

    private lateinit var hiddenInput: EditText
    private lateinit var statusDot: TextView
    private lateinit var statusText: TextView
    private var lastText = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        createUI()
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun createUI() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#121212"))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT)
        }

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(16), dp(48), dp(16), dp(16))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        val backBtn = TextView(this).apply {
            text = "←"
            textSize = 22f
            setTextColor(Color.WHITE)
            setPadding(0, 0, dp(16), 0)
            setOnClickListener { finish() }
        }

        val titleText = TextView(this).apply {
            text = "Remote Keyboard"
            textSize = 20f
            setTextColor(Color.WHITE)
            typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
        }

        topBar.addView(backBtn)
        topBar.addView(titleText)
        root.addView(topBar)

        val statusRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(24))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        statusDot = TextView(this).apply {
            text = "●"
            textSize = 10f
            setTextColor(
                if (ConnectionManager.isConnected()) Color.parseColor("#4CAF50")
                else Color.parseColor("#FF5555")
            )
            setPadding(0, 0, dp(8), 0)
        }

        statusText = TextView(this).apply {
            text = if (ConnectionManager.isConnected()) "Connected to PC" else "Not connected"
            textSize = 12f
            setTextColor(Color.parseColor("#888888"))
        }

        statusRow.addView(statusDot)
        statusRow.addView(statusText)
        root.addView(statusRow)

        val divider = LinearLayout(this).apply {
            setBackgroundColor(Color.parseColor("#222222"))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, dp(1)).apply {
                bottomMargin = dp(32)
            }
        }
        root.addView(divider)

        val centerArea = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(24), 0, dp(24), 0)
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        val hint = TextView(this).apply {
            text = "Tap the button below to open the keyboard.\nAnything you type will be sent to your PC."
            textSize = 13f
            setTextColor(Color.parseColor("#666666"))
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, dp(32))
        }
        centerArea.addView(hint)

        val typeBtn = TextView(this).apply {
            text = "⌨  Start Typing"
            textSize = 16f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(dp(40), dp(18), dp(40), dp(18))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dp(14).toFloat()
                setColor(Color.parseColor("#1E1E1E"))
            }
            layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply {
                bottomMargin = dp(32)
            }
            setOnClickListener { showKeyboard() }
        }
        centerArea.addView(typeBtn)

        val shortcutLabel = TextView(this).apply {
            text = "SHORTCUTS"
            textSize = 10f
            setTextColor(Color.parseColor("#444444"))
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
            letterSpacing = 0.1f
            setPadding(0, 0, 0, dp(14))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        centerArea.addView(shortcutLabel)

        val shortcutGrid = GridLayout(this).apply {
            columnCount = 3
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        shortcutGrid.addView(makeShortcutBtn("Enter") { sendKey("ENTER") })
        shortcutGrid.addView(makeShortcutBtn("Backspace") { sendKey("BACKSPACE") })
        shortcutGrid.addView(makeShortcutBtn("Tab") { sendKey("TAB") })
        shortcutGrid.addView(makeShortcutBtn("Esc") { sendKey("ESCAPE") })
        shortcutGrid.addView(makeShortcutBtn("↑") { sendKey("UP") })
        shortcutGrid.addView(makeShortcutBtn("Space") { sendKey("SPACE") })
        shortcutGrid.addView(makeShortcutBtn("←") { sendKey("LEFT") })
        shortcutGrid.addView(makeShortcutBtn("↓") { sendKey("DOWN") })
        shortcutGrid.addView(makeShortcutBtn("→") { sendKey("RIGHT") })
        shortcutGrid.addView(makeShortcutBtn("Ctrl+C") { sendKey("CTRL_C") })
        shortcutGrid.addView(makeShortcutBtn("Ctrl+V") { sendKey("CTRL_V") })
        shortcutGrid.addView(makeShortcutBtn("Ctrl+Z") { sendKey("CTRL_Z") })

        centerArea.addView(shortcutGrid)

        val comboLabel = TextView(this).apply {
            text = "KEY COMBINATIONS"
            textSize = 10f
            setTextColor(Color.parseColor("#444444"))
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
            letterSpacing = 0.1f
            setPadding(0, dp(24), 0, dp(14))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        centerArea.addView(comboLabel)

        val comboGrid = GridLayout(this).apply {
            columnCount = 3
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }

        comboGrid.addView(makeShortcutBtn("Task Manager") { sendKey("TASK_MANAGER") })
        comboGrid.addView(makeShortcutBtn("Ctrl+R") { sendKey("CTRL_R") })
        comboGrid.addView(makeShortcutBtn("Ctrl+S") { sendKey("CTRL_S") })
        comboGrid.addView(makeShortcutBtn("Ctrl+A") { sendKey("CTRL_A") })
        comboGrid.addView(makeShortcutBtn("Ctrl+X") { sendKey("CTRL_X") })
        comboGrid.addView(makeShortcutBtn("Ctrl+F") { sendKey("CTRL_F") })
        comboGrid.addView(makeShortcutBtn("Ctrl+W") { sendKey("CTRL_W") })
        comboGrid.addView(makeShortcutBtn("Ctrl+T") { sendKey("CTRL_T") })
        comboGrid.addView(makeShortcutBtn("Alt+F4") { sendKey("ALT_F4") })
        comboGrid.addView(makeShortcutBtn("Alt+Tab") { sendKey("ALT_TAB") })
        comboGrid.addView(makeShortcutBtn("Win+D") { sendKey("WIN_D") })
        comboGrid.addView(makeShortcutBtn("Win+R") { sendKey("WIN_R") })
        comboGrid.addView(makeShortcutBtn("Win+E") { sendKey("WIN_E") })
        comboGrid.addView(makeShortcutBtn("Ctrl+Shift+T") { sendKey("CTRL_SHIFT_T") })
        comboGrid.addView(makeShortcutBtn("Ctrl+Shift+Esc") { sendKey("CTRL_SHIFT_ESC") })

        centerArea.addView(comboGrid)
        root.addView(centerArea)

        hiddenInput = EditText(this).apply {
            layoutParams = LinearLayout.LayoutParams(1, 1)
            setBackgroundColor(Color.TRANSPARENT)
            setTextColor(Color.TRANSPARENT)
            isCursorVisible = false
            isSingleLine = false
        }
        root.addView(hiddenInput)

        hiddenInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val current = s.toString()
                if (current.length > lastText.length) {
                    val added = current.substring(lastText.length)
                    for (ch in added) {
                        ConnectionManager.send("KEY:CHAR:$ch")
                    }
                } else if (current.length < lastText.length) {
                    sendKey("BACKSPACE")
                }
                lastText = current
            }
        })

        hiddenInput.setOnKeyListener { _, keyCode, event ->
            if (event.action == KeyEvent.ACTION_DOWN) {
                when (keyCode) {
                    KeyEvent.KEYCODE_DEL -> sendKey("BACKSPACE")
                    KeyEvent.KEYCODE_ENTER -> sendKey("ENTER")
                    KeyEvent.KEYCODE_TAB -> sendKey("TAB")
                    KeyEvent.KEYCODE_ESCAPE -> sendKey("ESCAPE")
                    KeyEvent.KEYCODE_DPAD_UP -> sendKey("UP")
                    KeyEvent.KEYCODE_DPAD_DOWN -> sendKey("DOWN")
                    KeyEvent.KEYCODE_DPAD_LEFT -> sendKey("LEFT")
                    KeyEvent.KEYCODE_DPAD_RIGHT -> sendKey("RIGHT")
                }
            }
            false
        }

        setContentView(root)
    }

    private fun makeShortcutBtn(label: String, onClick: () -> Unit): LinearLayout {
        val btn = LinearLayout(this).apply {
            gravity = Gravity.CENTER
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dp(10).toFloat()
                setColor(Color.parseColor("#1E1E1E"))
            }
            isClickable = true
            isFocusable = true
            layoutParams = GridLayout.LayoutParams().apply {
                width = 0
                height = dp(52)
                columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1, 1f)
                setMargins(dp(4), dp(4), dp(4), dp(4))
            }
            setOnClickListener { onClick() }
        }

        val text = TextView(this).apply {
            text = label
            textSize = 13f
            setTextColor(Color.parseColor("#CCCCCC"))
            gravity = Gravity.CENTER
        }
        btn.addView(text)
        return btn
    }

    private fun showKeyboard() {
        hiddenInput.requestFocus()
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.showSoftInput(hiddenInput, InputMethodManager.SHOW_FORCED)
    }

    private fun sendKey(key: String) {
        ConnectionManager.send("KEY:$key")
    }
}