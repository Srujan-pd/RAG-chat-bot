variable "project_id" {
  type        = string
}

variable "region" {
  default = "us-central1"
  type    = string
}

variable "db_password" {
  type        = string
  sensitive   = true
}

variable "gemini_api_key" {
  type        = string
  sensitive   = true
}

variable "image_url" {
  type        = string
}
