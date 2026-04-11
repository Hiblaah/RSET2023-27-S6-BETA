-- =========================
-- USERS
-- =========================
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL
);

-- =========================
-- FILES
-- =========================
CREATE TABLE files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_size BIGINT NOT NULL,
    file_type VARCHAR(100),
    upload_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    num_chunks INT NOT NULL,

    INDEX idx_files_user (user_id),

    CONSTRAINT fk_files_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);

-- =========================
-- DEVICES
-- =========================
CREATE TABLE devices (
    device_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    device_name VARCHAR(255),
    device_fingerprint VARCHAR(255) NOT NULL UNIQUE,
    status ENUM('ONLINE', 'OFFLINE') NOT NULL DEFAULT 'OFFLINE',
    mode ENUM('User', 'Cluster') NOT NULL DEFAULT 'User', 
    last_heartbeat TIMESTAMP NOT NULL,
    storage_capacity BIGINT NOT NULL,
    available_storage BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_devices_user (user_id),
    INDEX idx_devices_status (status),
    INDEX idx_devices_mode (mode), -- Recommended for performance
    INDEX idx_devices_heartbeat (last_heartbeat),

    CONSTRAINT fk_devices_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE CASCADE
);

-- =========================
-- CHUNKS
-- =========================
CREATE TABLE chunks (
    chunk_id INT AUTO_INCREMENT PRIMARY KEY,
    file_id INT NOT NULL,
    chunk_index INT NOT NULL,
    chunk_hash VARCHAR(255) NOT NULL,
    chunk_size INT NOT NULL,

    UNIQUE KEY uniq_file_chunk (file_id, chunk_index),
    INDEX idx_chunks_file (file_id),

    CONSTRAINT fk_chunks_file
        FOREIGN KEY (file_id)
        REFERENCES files(file_id)
        ON DELETE CASCADE
);

-- =========================
-- CHUNK REPLICATION
-- =========================
CREATE TABLE chunk_replication (
    replication_id INT AUTO_INCREMENT PRIMARY KEY,

    chunk_id INT NOT NULL,
    device_id INT NOT NULL,

    replica_status ENUM('REPLICATING', 'ACTIVE', 'LOST','FAILED')
        NOT NULL DEFAULT 'REPLICATING',

    CONSTRAINT fk_chunk_replication_chunk
        FOREIGN KEY (chunk_id)
        REFERENCES chunks(chunk_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_chunk_replication_device
        FOREIGN KEY (device_id)
        REFERENCES devices(device_id)
        ON DELETE CASCADE,

    CONSTRAINT uq_chunk_device
        UNIQUE (chunk_id, device_id)
);

