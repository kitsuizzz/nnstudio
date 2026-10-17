# --- Build stage: compile the SPA ---
FROM node:22-alpine AS build
WORKDIR /app
# .dockerignore keeps node_modules/tmp/dist out of the context
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Turnstile site key for the sign-up "I'm not a robot" widget, baked in at
# build time (compose passes it; empty = no robot check in the SPA).
# .env is docker-ignored, so this is the only way the key reaches the build.
ARG VITE_TURNSTILE_SITE_KEY
ENV VITE_TURNSTILE_SITE_KEY=$VITE_TURNSTILE_SITE_KEY
# The mount path the SPA is built for: "/" (the docker stack at its own origin)
# or "/dev/" for the chat.kitsui.co/dev mount behind Caddy. Every asset and API
# URL derives from it, so it must match where the proxy serves the app.
ARG VITE_BASE_PATH=/
ENV VITE_BASE_PATH=$VITE_BASE_PATH
RUN npm run build

# --- Serve stage: nginx with same-origin API proxy ---
FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
