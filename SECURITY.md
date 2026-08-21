# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| v1.0.0 | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability in CrowdSafe AI, please report it responsibly.

**Do NOT open a public issue for security vulnerabilities.**

### How to Report

1. Email: [Your email here] or use GitHub's private vulnerability reporting
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- Acknowledgment within 48 hours
- Assessment within 1 week
- Fix timeline communicated promptly

## Security Considerations

### API Keys & Secrets
- Never commit `.env` files or API keys
- Use environment variables for all secrets
- Rotate keys regularly

### Model Security
- YOLO model files should be from trusted sources
- Verify model integrity before deployment

### Network Security
- FastAPI dashboard should use HTTPS in production
- Restrict access to authorized personnel only
- Use authentication for the web dashboard

### Data Privacy
- Video feeds may contain personally identifiable information
- Ensure compliance with local privacy regulations
- Implement data retention policies

## Scope

This security policy applies to the crowd_detection repository only.
