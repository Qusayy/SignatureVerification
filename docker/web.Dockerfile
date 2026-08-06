# Build the SPA, then serve it from nginx with the API proxied on the same
# origin. One origin means no CORS in production and no cookie-domain surprises
# between the demo and the production deployment.
#
# Build context is the repository root (not ./web) so nginx.conf is reachable.
FROM node:22-alpine AS build

WORKDIR /app
COPY web/package.json web/package-lock.json* ./
RUN npm ci || npm install

COPY web/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
