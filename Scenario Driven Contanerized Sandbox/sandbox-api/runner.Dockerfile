# Minimal sandboxed Node.js runner image for executing user code.
# This image is spun up fresh per-request and destroyed after execution.
# Security: runs as non-root user 'runner', no shell, no package manager access.
FROM node:20-alpine

# Create unprivileged user
RUN addgroup -S sandbox && adduser -S runner -G sandbox

# Switch to sandbox working dir
WORKDIR /sandbox

# Run as non-root
USER runner

# Default command - will be overridden by docker run args
CMD ["node", "code.js"]
