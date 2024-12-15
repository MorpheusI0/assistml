#!/bin/bash

MONGO_USER="admin"
MONGO_PASS="admin"
AUTH_DB="admin"
DB_NAME="assistml"
REPOSITORY_PATH="/repository"

for FILE in "$REPOSITORY_PATH"/*.json; do
  COLLECTION_NAME=$(basename "$FILE" .json)
  echo "Import $FILE to MongoDB to $COLLECTION_NAME..."

  while IFS= read -r LINE || [ -n "$LINE" ]; do
    echo "$LINE" | mongoimport \
      --username "$MONGO_USER" \
      --password "$MONGO_PASS" \
      --authenticationDatabase "$AUTH_DB" \
      --db "$DB_NAME" \
      --collection "$COLLECTION_NAME"
  done < "$FILE"
done
