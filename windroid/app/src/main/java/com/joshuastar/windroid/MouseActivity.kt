package com.joshuastar.windroid

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.*
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import kotlin.math.abs

class MouseActivity : Activity() {

    private var lastX = 0f
    private var lastY = 0f
    private var downX = 0f
    private var downY = 0f
    private var moved = false
    private var leftHeld = false
    private val MOVE_THRESHOLD = 8f

    private var lastScrollY = 0f
    private var scrolling = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0E0E15"))
        }

        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(18), dp(18), dp(18), dp(18))
            background = GradientDrawable(
                GradientDrawable.Orientation.LEFT_RIGHT,
                intArrayOf(Color.parseColor("#191927"), Color.parseColor("#12121A"))
            )
        }

        val backBtn = TextView(this).apply {
            text = "←"
            textSize = 22f
            setTextColor(Color.WHITE)
            setPadding(0, 0, dp(16), 0)
            setOnClickListener { finish() }
        }

        val title = TextView(this).apply {
            text = "Remote Mouse"
            textSize = 20f
            setTextColor(Color.WHITE)
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        }

        topBar.addView(backBtn)
        topBar.addView(title)

        val touchArea = View(this).apply {

            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
            ).apply {
                setMargins(dp(14), dp(14), dp(14), dp(14))
            }

            background = GradientDrawable().apply {
                cornerRadius = dp(18).toFloat()
                setColor(Color.parseColor("#1B1B2A"))
            }

            elevation = dp(6).toFloat()

            setOnTouchListener { _, event ->
                handleTouchArea(event)
                true
            }
        }

        val bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(14), dp(8), dp(14), dp(14))
        }

        fun createMouseButton(text: String): TextView {
            return TextView(this).apply {
                this.text = text
                gravity = Gravity.CENTER
                textSize = 16f
                setTextColor(Color.WHITE)
                background = GradientDrawable().apply {
                    cornerRadius = dp(16).toFloat()
                    setColor(Color.parseColor("#252538"))
                }
                elevation = dp(4).toFloat()
                layoutParams = LinearLayout.LayoutParams(
                    0,
                    dp(72),
                    1f
                ).apply {
                    marginEnd = dp(10)
                }
            }
        }

        val leftBtn = createMouseButton("Left Click")

        leftBtn.setOnTouchListener { _, event ->
            handleLeftButton(event)
            true
        }

        val rightBtn = createMouseButton("Right Click")

        rightBtn.setOnClickListener {
            ConnectionManager.send("MOUSE_RIGHT_CLICK")
        }

        bottomBar.addView(leftBtn)
        bottomBar.addView(rightBtn)

        root.addView(topBar)
        root.addView(touchArea)
        root.addView(bottomBar)

        setContentView(root)
    }

    private fun handleLeftButton(event: MotionEvent) {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                leftHeld = true
                ConnectionManager.send("MOUSE_DOWN")
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                leftHeld = false
                ConnectionManager.send("MOUSE_UP")
            }
        }
    }

    private fun handleTouchArea(event: MotionEvent) {

        if (event.pointerCount >= 2) {
            handleScroll(event)
            return
        }

        when (event.actionMasked) {

            MotionEvent.ACTION_DOWN -> {
                downX = event.x
                downY = event.y
                lastX = event.x
                lastY = event.y
                moved = false
            }

            MotionEvent.ACTION_MOVE -> {

                val dx = event.x - lastX
                val dy = event.y - lastY

                lastX = event.x
                lastY = event.y

                if (!moved && (abs(event.x - downX) > MOVE_THRESHOLD ||
                            abs(event.y - downY) > MOVE_THRESHOLD)) {
                    moved = true
                }

                if (moved) {
                    ConnectionManager.send("MOUSE_MOVE:${dx.toInt()},${dy.toInt()}")
                }
            }

            MotionEvent.ACTION_UP -> {
                if (!moved) {
                    ConnectionManager.send("MOUSE_CLICK")
                }
                moved = false
            }
        }
    }

    private fun handleScroll(event: MotionEvent) {

        when (event.actionMasked) {

            MotionEvent.ACTION_POINTER_DOWN -> {
                lastScrollY = event.getY(0)
                scrolling = true
            }

            MotionEvent.ACTION_MOVE -> {

                if (!scrolling) return

                val currentY = event.getY(0)
                val dy = currentY - lastScrollY
                lastScrollY = currentY

                ConnectionManager.send("MOUSE_SCROLL:${(-dy).toInt()}")
            }

            MotionEvent.ACTION_POINTER_UP -> {
                scrolling = false
            }
        }
    }
}