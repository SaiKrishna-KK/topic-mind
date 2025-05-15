#!/usr/bin/env python3
"""
Organizes TopicMind log files into a structured folder hierarchy.
Run this script to clean up and organize log files after testing.
"""

import os
import shutil
import glob
import datetime
import json
from pathlib import Path

# Main logs directory
LOGS_ROOT = "logs"

# Output organized structure
ORGANIZED_LOGS = {
    "gpt": {
        "validation": "GPT evaluation and validation results",
        "feedback": "GPT-generated feedback on summaries"
    },
    "semantic": {
        "embeddings": "Semantic embedding data",
        "cleaned": "Cleaned output from semantic processing"
    },
    "summaries": {
        "final": "Final generated summaries",
        "chunks": "Individual chunk summaries",
        "raw": "Raw summary data before post-processing"
    },
    "eval": {
        "reports": "Summary evaluation reports",
        "metrics": "Quality metrics and scores"
    },
    "pipeline": {
        "reports": "Detailed pipeline processing reports"
    },
    "tests": {
        "batch_tests": "Results from batch test runs",
        "individual_tests": "Individual test results"
    },
    "streamlit": {
        "user_sessions": "User session data from Streamlit",
        "exports": "Exported summaries from UI"
    },
    "archived": {
        "dated_archives": "Archived logs organized by date"
    }
}

def setup_log_folders():
    """Creates the organized log folder structure"""
    print("Setting up organized log folder structure...")
    
    for category, subfolders in ORGANIZED_LOGS.items():
        for subfolder, description in subfolders.items():
            # Create complete path
            folder_path = os.path.join(LOGS_ROOT, category, subfolder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create a README in each folder explaining its purpose
            readme_path = os.path.join(folder_path, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write(f"# {subfolder.replace('_', ' ').title()}\n\n")
                    f.write(f"{description}\n\n")
                    f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
    
    print("Folder structure created successfully.")

def organize_pipeline_logs():
    """Organizes existing pipeline logs into the new structure"""
    print("Organizing pipeline logs...")
    
    # Define source patterns and target folders
    log_patterns = [
        # GPT Validation logs
        {
            "pattern": "logs/gpt_validation_*.json",
            "destination": os.path.join(LOGS_ROOT, "gpt", "validation")
        },
        # Summary feedback logs 
        {
            "pattern": "logs/pipeline/summary_feedback_*.json",
            "destination": os.path.join(LOGS_ROOT, "gpt", "feedback")
        },
        # Semantic processing logs
        {
            "pattern": "logs/semantic_cleaned_*.txt",
            "destination": os.path.join(LOGS_ROOT, "semantic", "cleaned")
        },
        # Chunk pass logs
        {
            "pattern": "logs/pipeline/chunk_pass_log_*.json",
            "destination": os.path.join(LOGS_ROOT, "summaries", "chunks")
        },
        # Final summary logs
        {
            "pattern": "logs/pipeline/final_summary_pass_*.txt",
            "destination": os.path.join(LOGS_ROOT, "summaries", "final")
        },
        # Evaluation reports
        {
            "pattern": "logs/eval_report_*.json",
            "destination": os.path.join(LOGS_ROOT, "eval", "reports")
        },
        # Enhanced pipeline results
        {
            "pattern": "logs/pipeline/enhanced_pipeline_result_*.json",
            "destination": os.path.join(LOGS_ROOT, "pipeline", "reports")
        },
        # Human-readable reports
        {
            "pattern": "logs/pipeline/enhanced_pipeline_report_*.txt",
            "destination": os.path.join(LOGS_ROOT, "pipeline", "reports")
        },
        # Test logs
        {
            "pattern": "logs/test_thread_pipeline_*.log",
            "destination": os.path.join(LOGS_ROOT, "tests", "batch_tests")
        }
    ]
    
    # Process each pattern
    for pattern_info in log_patterns:
        pattern = pattern_info["pattern"]
        destination = pattern_info["destination"]
        
        # Find matching files
        matching_files = glob.glob(pattern)
        print(f"Found {len(matching_files)} files matching {pattern}")
        
        # Copy files to organized location
        for file_path in matching_files:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(destination, filename)
            
            # If the file has a timestamp, create a dated subfolder
            if "_20" in filename:  # Look for timestamp pattern like _20250515
                try:
                    # Try to parse timestamp from filename
                    timestamp_part = filename.split("_")[1][:8]  # e.g., 20250515
                    if len(timestamp_part) == 8 and timestamp_part.isdigit():
                        # Create YYYY-MM-DD format
                        date_folder = f"{timestamp_part[:4]}-{timestamp_part[4:6]}-{timestamp_part[6:8]}"
                        dated_destination = os.path.join(destination, date_folder)
                        os.makedirs(dated_destination, exist_ok=True)
                        dest_path = os.path.join(dated_destination, filename)
                except (IndexError, ValueError):
                    pass  # Use default destination if timestamp extraction fails
            
            # Copy the file
            if os.path.exists(file_path) and not os.path.exists(dest_path):
                shutil.copy2(file_path, dest_path)
                print(f"  Copied: {file_path} → {dest_path}")

def create_log_index():
    """Creates an index of all log files for easy browsing"""
    print("Creating log index...")
    
    index_data = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "categories": {}
    }
    
    # Walk through the logs directory structure
    for root, dirs, files in os.walk(LOGS_ROOT):
        # Skip the root directory itself
        if root == LOGS_ROOT:
            continue
            
        # Get relative path to use as category
        rel_path = os.path.relpath(root, LOGS_ROOT)
        
        # Skip if it's just a directory with only a README
        if len(files) <= 1 and "README.md" in files:
            continue
            
        # Skip hidden directories
        if any(part.startswith('.') for part in rel_path.split(os.path.sep)):
            continue
            
        # Add this category to our index
        if rel_path not in index_data["categories"]:
            index_data["categories"][rel_path] = []
            
        # Add files to this category (excluding READMEs)
        for file in sorted(files):
            if file != "README.md" and not file.startswith('.'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                file_mtime = datetime.datetime.fromtimestamp(
                    os.path.getmtime(file_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
                
                index_data["categories"][rel_path].append({
                    "name": file,
                    "path": os.path.join(rel_path, file),
                    "size": file_size,
                    "modified": file_mtime
                })
    
    # Write index as JSON
    index_path = os.path.join(LOGS_ROOT, "log_index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
        
    # Write a human-readable version
    index_md_path = os.path.join(LOGS_ROOT, "log_index.md")
    with open(index_md_path, "w") as f:
        f.write("# TopicMind Log Index\n\n")
        f.write(f"Last updated: {index_data['last_updated']}\n\n")
        
        for category, files in index_data["categories"].items():
            if not files:
                continue
                
            f.write(f"## {category.replace('/', ' › ')}\n\n")
            
            f.write("| File | Modified | Size |\n")
            f.write("|------|----------|------|\n")
            
            for file_info in files:
                # Format file size nicely
                size_kb = file_info["size"] / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
                
                f.write(f"| {file_info['name']} | {file_info['modified']} | {size_str} |\n")
            
            f.write("\n")
    
    print(f"Created log index at {index_path} and {index_md_path}")

def archive_old_logs(days_old=30):
    """Archives logs older than the specified number of days"""
    print(f"Archiving logs older than {days_old} days...")
    
    # Get current time
    now = datetime.datetime.now()
    cutoff = now - datetime.timedelta(days=days_old)
    
    # Create archive folder with timestamp
    archive_date = now.strftime("%Y%m%d")
    archive_dir = os.path.join(LOGS_ROOT, "archived", "dated_archives", f"archive_{archive_date}")
    os.makedirs(archive_dir, exist_ok=True)
    
    # Find all log files
    all_logs = []
    for root, _, files in os.walk(LOGS_ROOT):
        for file in files:
            if file.endswith(('.log', '.json', '.txt')) and not file.startswith('.'):
                file_path = os.path.join(root, file)
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if mtime < cutoff:
                    all_logs.append((file_path, mtime))
    
    # Archive old logs
    archived_count = 0
    for file_path, mtime in all_logs:
        # Don't archive files that are already in an archive
        if "archived" in file_path:
            continue
            
        # Get relative path to maintain folder structure in archive
        rel_path = os.path.relpath(file_path, LOGS_ROOT)
        archive_path = os.path.join(archive_dir, rel_path)
        
        # Create directory structure in archive
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        
        # Move file to archive
        if os.path.exists(file_path):
            shutil.move(file_path, archive_path)
            archived_count += 1
    
    print(f"Archived {archived_count} old log files to {archive_dir}")
    
    # Create a manifest file
    manifest_path = os.path.join(archive_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"TopicMind Log Archive Created: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Contains logs older than: {cutoff.strftime('%Y-%m-%d')}\n")
        f.write(f"Total files archived: {archived_count}\n\n")
        f.write("Archive Contents:\n")
        
        # List all files in the archive
        for root, _, files in os.walk(archive_dir):
            if root == archive_dir:
                continue
            rel_root = os.path.relpath(root, archive_dir)
            f.write(f"\n{rel_root}/\n")
            for file in sorted(files):
                if file != "manifest.txt":
                    f.write(f"  - {file}\n")

def main():
    """Main function to organize logs"""
    print("TopicMind Log Organizer")
    print("======================")
    
    # Ensure main logs directory exists
    os.makedirs(LOGS_ROOT, exist_ok=True)
    
    # Set up the organized folder structure
    setup_log_folders()
    
    # Organize existing logs
    organize_pipeline_logs()
    
    # Create a log index for easy browsing
    create_log_index()
    
    # Optionally archive old logs (disabled by default)
    # Uncomment to enable:
    # archive_old_logs(days_old=30)
    
    print("\nLog organization complete.")
    print(f"Logs are now organized in the {LOGS_ROOT}/ directory.")

if __name__ == "__main__":
    main() 