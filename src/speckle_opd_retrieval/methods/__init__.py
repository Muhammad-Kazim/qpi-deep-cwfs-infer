"""One subpackage per OPD retrieval method. Methods are imported lazily via the registry
(``speckle_opd_retrieval.get_method``), so importing this package never requires torch."""
