"""
Scheduled Task Manager for Automated Model Retraining
====================================================

This module provides scheduled execution of the automated model retraining pipeline.
It can be run as a standalone service or integrated with cron jobs.
"""

import time
import schedule
import logging
from datetime import datetime, timedelta
from model_retraining import run_retraining_pipeline
import threading
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetrainingScheduler:
    """Scheduler for automated model retraining"""
    
    def __init__(self):
        self.running = False
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        
    def run_scheduled_retraining(self):
        """Execute scheduled retraining with logging"""
        logger.info("🔄 Starting scheduled retraining check...")
        
        try:
            result = run_retraining_pipeline(force=False)
            
            if result.get('retraining_performed'):
                logger.info("✅ Scheduled retraining completed successfully")
                logger.info(f"📊 Result: {result.get('result', {}).get('deployment_status', 'Unknown')}")
            else:
                logger.info("ℹ️ No retraining needed at this time")
                
        except Exception as e:
            logger.error(f"❌ Scheduled retraining failed: {str(e)}")
            
        logger.info("🏁 Scheduled retraining check completed")
    
    def start_scheduler(self, check_frequency_hours=24):
        """Start the retraining scheduler"""
        logger.info(f"🚀 Starting retraining scheduler (checking every {check_frequency_hours} hours)")
        
        # Schedule the retraining check
        schedule.every(check_frequency_hours).hours.do(self.run_scheduled_retraining)
        
        # Also run daily at 2 AM
        schedule.every().day.at("02:00").do(self.run_scheduled_retraining)
        
        # Run once immediately to check status
        self.run_scheduled_retraining()
        
        self.running = True
        
        def scheduler_loop():
            """Main scheduler loop"""
            while self.running and not self.stop_event.is_set():
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
            
            logger.info("📴 Scheduler stopped")
        
        # Start scheduler in background thread
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Retraining scheduler started successfully")
    
    def stop_scheduler(self):
        """Stop the retraining scheduler"""
        logger.info("🛑 Stopping retraining scheduler...")
        self.running = False
        self.stop_event.set()
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10)
        
        schedule.clear()
        logger.info("✅ Retraining scheduler stopped")
    
    def get_status(self):
        """Get scheduler status"""
        next_runs = []
        for job in schedule.jobs:
            next_runs.append({
                'job': str(job.job_func),
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'interval': str(job.interval)
            })
        
        return {
            'running': self.running,
            'scheduled_jobs': len(schedule.jobs),
            'next_runs': next_runs,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }

# Global scheduler instance
scheduler = RetrainingScheduler()

def start_background_scheduler():
    """Start the background scheduler"""
    if not scheduler.running:
        scheduler.start_scheduler()
        return True
    return False

def stop_background_scheduler():
    """Stop the background scheduler"""
    if scheduler.running:
        scheduler.stop_scheduler()
        return True
    return False

def get_scheduler_status():
    """Get current scheduler status"""
    return scheduler.get_status()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    stop_background_scheduler()
    sys.exit(0)

def main():
    """Main function for running scheduler as standalone service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Model Retraining Scheduler")
    parser.add_argument("--hours", type=int, default=24, help="Check frequency in hours")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon service")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--stop", action="store_true", help="Stop running scheduler")
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.status:
        status = get_scheduler_status()
        print(f"Scheduler Status: {'Running' if status['running'] else 'Stopped'}")
        print(f"Scheduled Jobs: {status['scheduled_jobs']}")
        if status['next_runs']:
            print("Next Runs:")
            for run in status['next_runs']:
                print(f"  - {run['job']}: {run['next_run']}")
        return
    
    if args.stop:
        if stop_background_scheduler():
            print("Scheduler stopped successfully")
        else:
            print("Scheduler was not running")
        return
    
    try:
        # Start the scheduler
        scheduler.start_scheduler(check_frequency_hours=args.hours)
        
        if args.daemon:
            # Run as daemon - keep running indefinitely
            logger.info("Running as daemon service...")
            try:
                while scheduler.running:
                    time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
        else:
            # Interactive mode
            print("🔄 Automated Retraining Scheduler Started")
            print("⏰ Checking for retraining needs periodically...")
            print("🔍 Press Ctrl+C to stop")
            
            try:
                while scheduler.running:
                    status = scheduler.get_status()
                    next_run = None
                    if status['next_runs']:
                        next_run = status['next_runs'][0]['next_run']
                    
                    print(f"\r⏰ Scheduler running... Next check: {next_run or 'Unknown'}", end='', flush=True)
                    time.sleep(30)
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping scheduler...")
    
    finally:
        stop_background_scheduler()
        print("✅ Scheduler stopped successfully")

if __name__ == "__main__":
    main()