## 📋 Pull Request Checklist

### Description
<!-- Provide a brief description of the changes in this PR -->

**What does this PR do?**
- [ ] 🆕 Adds new feature
- [ ] 🐛 Fixes bug
- [ ] 📚 Updates documentation
- [ ] 🧹 Refactors existing code
- [ ] ⚡ Improves performance
- [ ] 🔧 Updates configuration
- [ ] 🧪 Adds/updates tests

**Related Issue(s):**
<!-- Link to related issues using # -->
Closes #

### Changes Made
<!-- Describe the specific changes made in this PR -->

### Testing
<!-- Describe how you tested your changes -->

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] E2E tests pass
- [ ] Performance impact assessed

**Test Command:**
```bash
make test
```

### Code Quality
<!-- Confirm code quality checks -->

- [ ] Code follows PEP 8 style guidelines
- [ ] Code is properly documented
- [ ] No new linting errors introduced
- [ ] Type hints added where appropriate
- [ ] Security considerations addressed

**Quality Checks:**
```bash
make format
make lint
make type-check
```

### API Changes
<!-- If this PR includes API changes -->

- [ ] No breaking API changes
- [ ] API documentation updated
- [ ] Backward compatibility maintained
- [ ] OpenAPI schema updated

### Database Changes
<!-- If this PR includes database changes -->

- [ ] Migration scripts included
- [ ] Database schema documented
- [ ] Data migration tested
- [ ] Rollback procedure documented

### Dependencies
<!-- If this PR includes dependency changes -->

- [ ] New dependencies justified
- [ ] Security scan passed
- [ ] License compatibility verified
- [ ] pyproject.toml updated

### Documentation
<!-- Confirm documentation is updated -->

- [ ] README updated (if needed)
- [ ] API docs updated (if needed)
- [ ] Code comments added
- [ ] Configuration changes documented

### Deployment
<!-- For production-impacting changes -->

- [ ] Docker build tested
- [ ] Environment variables updated
- [ ] Configuration changes documented
- [ ] Deployment plan reviewed

### Review Requests
<!-- Tag specific reviewers if needed -->

**Required Reviewers:**
- [ ] Technical Lead (for architecture changes)
- [ ] Data Engineer (for data pipeline changes)
- [ ] DevOps (for infrastructure changes)

**Reviewer Guidelines:**
Please review for:
- Code correctness and logic
- Test coverage and quality
- Performance implications
- Security considerations
- Documentation completeness
- API design and compatibility

### Screenshots/Demo
<!-- Add screenshots or demo links if applicable -->

### Performance Impact
<!-- Describe any performance implications -->

- [ ] No performance impact
- [ ] Positive performance impact
- [ ] Negative performance impact (justified below)

**Performance Notes:**
<!-- Add performance benchmarks or analysis -->

### Security Review
<!-- For security-sensitive changes -->

- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Authentication/authorization verified
- [ ] Secrets properly managed

### Additional Notes
<!-- Any additional information for reviewers -->

---

### Reviewer Checklist
<!-- For reviewers to complete -->

#### Functionality Review
- [ ] Changes work as described
- [ ] Edge cases considered
- [ ] Error handling is appropriate
- [ ] Business logic is correct

#### Code Quality Review
- [ ] Code is readable and maintainable
- [ ] Follows project conventions
- [ ] No code duplication
- [ ] Appropriate abstractions used

#### Testing Review
- [ ] Tests cover the changes
- [ ] Tests are reliable and fast
- [ ] Test data is appropriate
- [ ] Integration points tested

#### Documentation Review
- [ ] Code is self-documenting
- [ ] Complex logic explained
- [ ] API changes documented
- [ ] User-facing changes documented

#### Security Review
- [ ] No security vulnerabilities introduced
- [ ] Input validation sufficient
- [ ] Output encoding appropriate
- [ ] Access controls maintained

---

**Merge Criteria:**
- [ ] All checks pass
- [ ] At least 2 approvals
- [ ] No unresolved conversations
- [ ] Up to date with target branch