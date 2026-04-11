package com.phonecluster.app.utils

import org.bouncycastle.crypto.InvalidCipherTextException
import org.bouncycastle.crypto.engines.AsconAEAD128
import org.bouncycastle.crypto.params.AEADParameters
import org.bouncycastle.crypto.params.KeyParameter
import java.security.MessageDigest
import java.security.SecureRandom

object AsconFileCrypto {
    private const val KEY_SIZE = 16
    private const val NONCE_SIZE = 16
    private const val MAC_SIZE_BITS = 128
    private val rng = SecureRandom()
    private val MAGIC = byteArrayOf(0x41, 0x53, 0x43, 0x4E)

    fun encryptFile(
        plaintext: ByteArray,
        password: String,
        aad: ByteArray = byteArrayOf()
    ): ByteArray {
        val key = deriveKey(password)
        val nonce = ByteArray(NONCE_SIZE).also { rng.nextBytes(it) }

        val cipher = AsconAEAD128()
        val params = AEADParameters(KeyParameter(key), MAC_SIZE_BITS, nonce, aad)
        cipher.init(true, params)

        val out = ByteArray(cipher.getOutputSize(plaintext.size))
        var len = cipher.processBytes(plaintext, 0, plaintext.size, out, 0)
        len += cipher.doFinal(out, len)

        return MAGIC + nonce + out.copyOf(len)
    }

    fun decryptFile(
        encryptedFile: ByteArray,
        password: String,
        aad: ByteArray = byteArrayOf()
    ): ByteArray {
        require(encryptedFile.size >= 4 + NONCE_SIZE + 16) { "Encrypted file too small" }

        val magic = encryptedFile.copyOfRange(0, 4)
        require(magic.contentEquals(MAGIC)) { "Invalid ASCON file format" }

        val nonce = encryptedFile.copyOfRange(4, 4 + NONCE_SIZE)
        val cipherTextWithTag = encryptedFile.copyOfRange(4 + NONCE_SIZE, encryptedFile.size)

        val key = deriveKey(password)

        val cipher = AsconAEAD128()
        val params = AEADParameters(KeyParameter(key), MAC_SIZE_BITS, nonce, aad)
        cipher.init(false, params)

        val out = ByteArray(cipher.getOutputSize(cipherTextWithTag.size))
        var len = cipher.processBytes(cipherTextWithTag, 0, cipherTextWithTag.size, out, 0)

        try {
            len += cipher.doFinal(out, len)
        } catch (e: InvalidCipherTextException) {
            throw IllegalArgumentException("Wrong password or file tampered")
        }

        return out.copyOf(len)
    }

    private fun deriveKey(password: String): ByteArray {
        return MessageDigest.getInstance("SHA-256")
            .digest(password.toByteArray(Charsets.UTF_8))
            .copyOf(KEY_SIZE)
    }
}