import { useState, useEffect, useRef } from "react";
import axios from "axios";

export default function FighterSelect({ corner, selected, onSelect, api }) {
  const [query, setQuery]     = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen]       = useState(false);
  const ref = useRef();

  useEffect(() => {
    if (query.length < 2) { setResults([]); setOpen(false); return; }
    const timer = setTimeout(async () => {
      try {
        const { data } = await axios.get(`${api}/fighters/search`, { params: { q: query } });
        setResults(data);
        setOpen(true);
      } catch (e) {
        console.error("Search error:", e);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [query, api]);

  useEffect(() => {
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleSelect = async (name) => {
    if (!name || !name.trim()) return;
    try {
      console.log("Fetching fighter:", name);
      const { data } = await axios.get(`${api}/fighters/${encodeURIComponent(name)}`);
      onSelect(data);
      setQuery(name);
      setOpen(false);
      setResults([]);
    } catch (e) {
      console.error("Fighter fetch error:", e.response?.status, name);
    }
  };

  const borderColor = corner === "red" ? "#ef4444" : "#3b82f6";

  return (
    <div className="fighter-select" ref={ref}>
      <input
        className="search-input"
        style={{ borderColor: selected ? borderColor : undefined }}
        placeholder="Search fighter name..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          if (selected) onSelect(null);
        }}
        onFocus={() => results.length > 0 && setOpen(true)}
      />
      {open && results.length > 0 && (
        <ul className="dropdown">
          {results.map((name) => (
            <li
              key={name}
              onMouseDown={(e) => {
                e.preventDefault();
                handleSelect(name);
              }}
              className="dropdown-item"
            >
              {name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}