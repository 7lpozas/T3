const RATE_LIMIT = 10; // 10 requests per minute
const INTERVAL = 60 * 1000; // 1 minute in milliseconds

class RateLimiter {
  constructor() {
    this.requests = [];
  }

  canMakeRequest() {
    const now = Date.now();
    this.requests = this.requests.filter(time => now - time < INTERVAL);
    if (this.requests.length < RATE_LIMIT) {
      this.requests.push(now);
      return true;
    }
    return false;
  }
}

export default new RateLimiter();