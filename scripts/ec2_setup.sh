#!/bin/bash
set -euo pipefail

AWS_REGION="eu-central-1"
ECR_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
ECR_REPO="battlesnake"

echo "==> Installing dependencies"
sudo dnf update -y
sudo dnf install -y docker nginx curl unzip

echo "==> Starting Docker"
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

echo "==> Installing AWS CLI v2"
curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
sudo /tmp/aws/install

echo "==> Pulling initial image from ECR"
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    "${ECR_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker pull "${ECR_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
docker run -d --name battlesnake \
  -p 8000:8000 \
  --restart unless-stopped \
  "${ECR_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

echo "==> Configuring Nginx"
sudo cp ~/battlesnake.nginx.conf /etc/nginx/conf.d/battlesnake.conf
sudo nginx -t
sudo systemctl enable --now nginx

echo "==> Done. Snake is live on port 80."
