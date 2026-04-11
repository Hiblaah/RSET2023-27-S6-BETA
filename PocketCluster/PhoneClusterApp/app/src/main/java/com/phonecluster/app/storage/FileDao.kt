package com.phonecluster.app.storage

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow
@Dao
interface FileDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(file: FileEntity)

    @Query("SELECT * FROM files")
    fun getAllFiles(): Flow<List<FileEntity>>

    @Query("SELECT * FROM files")
    suspend fun getAllFilesOnce(): List<FileEntity>

    @Query("SELECT * FROM files WHERE serverFileId = :id LIMIT 1")
    suspend fun getFileByServerId(id: Long): FileEntity?

    @Query("DELETE FROM files WHERE serverFileId = :id")
    suspend fun deleteFileByServerId(id: Long)
}
