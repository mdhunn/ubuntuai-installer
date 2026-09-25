PREFIX ?= /usr/local
DESTDIR ?=
PYTHON ?= python3
HELPER = $(PREFIX)/sbin/ubuntuai-installer-helper
POLICY_IN = usr/share/polkit-1/actions/org.ubuntuai.pkexec.policy.in
POLICY_OUT = usr/share/polkit-1/actions/org.ubuntuai.pkexec.policy
POLKIT_DIR = /usr/share/polkit-1/actions

.PHONY: test install uninstall policy catalog-drift

test:
	$(PYTHON) -m unittest discover -s tests -v

catalog-drift:
	$(PYTHON) usr/share/ubuntuai-installer/catalog_drift.py

policy:
	sed 's|@HELPER@|$(HELPER)|g' $(POLICY_IN) > $(POLICY_OUT)

install: policy
	install -d $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/sbin
	install -d $(DESTDIR)$(PREFIX)/share/ubuntuai-installer/ui
	install -d $(DESTDIR)$(PREFIX)/share/ubuntuai-installer/limits
	install -d $(DESTDIR)$(PREFIX)/share/applications
	install -d $(DESTDIR)$(PREFIX)/share/icons/hicolor/scalable/apps
	install -d $(DESTDIR)$(PREFIX)/share/man/man1
	install -d $(DESTDIR)$(POLKIT_DIR)
	install -m 0755 usr/bin/ubuntuai-installer $(DESTDIR)$(PREFIX)/bin/ubuntuai-installer
	install -m 0755 usr/bin/ubuntuai-config $(DESTDIR)$(PREFIX)/bin/ubuntuai-config
	install -m 0755 usr/bin/ubuntuai-validate $(DESTDIR)$(PREFIX)/bin/ubuntuai-validate
	install -m 0755 usr/sbin/ubuntuai-installer-helper $(DESTDIR)$(PREFIX)/sbin/ubuntuai-installer-helper
	install -m 0644 usr/share/ubuntuai-installer/*.py $(DESTDIR)$(PREFIX)/share/ubuntuai-installer/
	install -m 0644 usr/share/ubuntuai-installer/*.json $(DESTDIR)$(PREFIX)/share/ubuntuai-installer/
	install -m 0644 usr/share/ubuntuai-installer/ui/*.py $(DESTDIR)$(PREFIX)/share/ubuntuai-installer/ui/
	install -m 0644 usr/share/ubuntuai-installer/limits/30-ubuntuai.conf $(DESTDIR)$(PREFIX)/share/ubuntuai-installer/limits/
	install -m 0644 usr/share/applications/*.desktop $(DESTDIR)$(PREFIX)/share/applications/
	install -m 0644 usr/share/icons/hicolor/scalable/apps/ubuntuai-installer.svg $(DESTDIR)$(PREFIX)/share/icons/hicolor/scalable/apps/
	install -m 0644 usr/share/man/man1/*.1 $(DESTDIR)$(PREFIX)/share/man/man1/
	install -m 0644 $(POLICY_OUT) $(DESTDIR)$(POLKIT_DIR)/org.ubuntuai.pkexec.policy
	-gtk-update-icon-cache -f $(DESTDIR)$(PREFIX)/share/icons/hicolor 2>/dev/null
	-update-desktop-database $(DESTDIR)$(PREFIX)/share/applications 2>/dev/null

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/ubuntuai-installer
	rm -f $(DESTDIR)$(PREFIX)/bin/ubuntuai-config
	rm -f $(DESTDIR)$(PREFIX)/bin/ubuntuai-validate
	rm -f $(DESTDIR)$(PREFIX)/sbin/ubuntuai-installer-helper
	rm -rf $(DESTDIR)$(PREFIX)/share/ubuntuai-installer
	rm -f $(DESTDIR)$(PREFIX)/share/applications/ubuntuai-installer.desktop
	rm -f $(DESTDIR)$(PREFIX)/share/applications/ubuntuai-config.desktop
	rm -f $(DESTDIR)$(PREFIX)/share/icons/hicolor/scalable/apps/ubuntuai-installer.svg
	rm -f $(DESTDIR)$(PREFIX)/share/man/man1/ubuntuai-installer.1
	rm -f $(DESTDIR)$(PREFIX)/share/man/man1/ubuntuai-config.1
	rm -f $(DESTDIR)$(PREFIX)/share/man/man1/ubuntuai-validate.1
	rm -f $(DESTDIR)$(POLKIT_DIR)/org.ubuntuai.pkexec.policy
