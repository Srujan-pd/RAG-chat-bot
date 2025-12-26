terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_sql_database_instance" "instance" {
  name             = "chatbot-db-instance"
  region           = var.region
  database_version = "POSTGRES_15"
  settings {
    tier = "db-f1-micro" # Smallest tier for cost efficiency
  }
  deletion_protection = false
}

resource "google_sql_database" "database" {
  name     = "genai"
  instance = google_sql_database_instance.instance.name
}

resource "google_sql_user" "db_user" {
  name     = "postgres"
  instance = google_sql_database_instance.instance.name
  password = var.db_password
}

resource "google_cloud_run_v2_service" "chatbot" {
  name     = "genai-chatbot"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = var.image_url
      
      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }
      
      env {
        name  = "DATABASE_URL"
        value = "postgresql+psycopg2://postgres:${var.db_password}@/genai?host=/cloudsql/${google_sql_database_instance.instance.connection_name}"
      }

      ports {
        container_port = 8080
      }
    }

    annotations = {
      "run.googleapis.com/cloudsql-instances" = google_sql_database_instance.instance.connection_name
    }
  }
}

resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.chatbot.location
  service  = google_cloud_run_v2_service.chatbot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
