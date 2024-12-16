#!/bin/bash

MYSQL_CONF_FILE="/etc/mysql/my.cnf"

if ! grep -q "bind-address" "$MYSQL_CONF_FILE"; then
  echo "Adding bind-address configuration to $MYSQL_CONF_FILE..."
  echo "[mysqld]" >> "$MYSQL_CONF_FILE"
  echo "bind-address = 0.0.0.0" >> "$MYSQL_CONF_FILE"
fi

if [ ! -d "$MYSQL_DATA_DIR/mysql" ]; then
  echo "Initializing MySQL data directory..."
  mysqld --initialize-insecure --user=mysql
  chown -R mysql:mysql $MYSQL_DATA_DIR
  echo "MySQL data directory initialized."
else
  echo "MySQL data directory already exists. Skipping initialization."
fi

function sync_to_s3_on_exit {
    echo "Uploading data to S3 before container exits..."
    mysqldump -u root --password="$MYSQL_ROOT_PASSWORD" meowmung > "$MYSQL_DATA_DIR/$BACKUP_FILE"
    aws s3 cp "$MYSQL_DATA_DIR/$BACKUP_FILE" "$S3_BUCKET$BACKUP_FILE"
    echo "Data upload to S3 completed."
}

trap sync_to_s3_on_exit EXIT

echo "Starting MySQL server..."
mysqld --user=mysql &

sleep 5

mysqladmin -u root password "$MYSQL_ROOT_PASSWORD"
echo "Root password set to $MYSQL_ROOT_PASSWORD."

echo "Downloading backup.sql from S3..."
aws s3 cp "$S3_BUCKET$BACKUP_FILE" "$MYSQL_DATA_DIR/$BACKUP_FILE"
echo "Backup file downloaded."

echo "Restoring database from backup..."
mysql -u root --password="$MYSQL_ROOT_PASSWORD" -e "CREATE DATABASE IF NOT EXISTS meowmung;"
mysql -u root --password="$MYSQL_ROOT_PASSWORD" -e "CREATE USER 'meowmung'@'%' IDENTIFIED BY 'meowmung1234';"
mysql -u root --password="$MYSQL_ROOT_PASSWORD" -e "GRANT ALL PRIVILEGES ON *.* TO 'meowmung'@'%';"
mysql -u root --password="$MYSQL_ROOT_PASSWORD" -e "FLUSH PRIVILEGES;"
mysql -u root --password="$MYSQL_ROOT_PASSWORD" meowmung < "$MYSQL_DATA_DIR/$BACKUP_FILE"
echo "Database restoration completed."

wait
