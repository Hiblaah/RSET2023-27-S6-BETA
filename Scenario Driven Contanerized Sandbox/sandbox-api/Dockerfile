# Dockerfile for the Sandbox API server itself
FROM node:20-alpine

WORKDIR /app

# Install dependencies and Docker CLI
RUN apk add --no-cache docker-cli
COPY package*.json ./
RUN npm install --omit=dev

# Copy source
COPY . .

EXPOSE 3001

CMD ["node", "server.js"]
